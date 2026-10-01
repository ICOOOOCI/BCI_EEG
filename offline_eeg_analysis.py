"""在真实会话上重放解码，并按完整 block 验证个人模板；不启动 UI/LSL。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields, replace
import hashlib
import html
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import time
from types import SimpleNamespace
import zipfile

import fbcca_keyboard_dual_mode as decoder
import numpy as np


def load_session(path):
    """仅读取 JSON 和非对象 NPZ，不执行会话中的 pickle。"""
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('_metadata.json')]
        if len(names) != 1:
            raise ValueError('需要一个包含完整元数据的会话 ZIP')
        manifest = json.loads(archive.read(names[0]))
        stem = names[0][:-len('_metadata.json')]
        with np.load(io.BytesIO(archive.read(stem + '.npz')), allow_pickle=False) as data:
            session = json.loads(data['metadata_json'].item())
            trials = [t for t in session['trials'] if t['status'] == 'valid'
                      and t['true_class'] is not None]
            windows = np.array([data[f"trial_{t['trial_id']:04d}_uniform_data"] for t in trials])
            scores = np.array([data[f"trial_{t['trial_id']:04d}_scores"] for t in trials])
            grids = np.array([data[f"trial_{t['trial_id']:04d}_uniform_timestamps"] for t in trials])
        with np.load(io.BytesIO(archive.read(stem + '_continuous_eeg.npz')), allow_pickle=False) as data:
            all_channels = data['source_data']
            constant_channels = np.flatnonzero(np.ptp(all_channels, axis=-1) == 0).tolist()
            continuous = all_channels[manifest['config']['channel_indices']].astype(float)
            timestamps = data['lag_corrected_lsl_timestamps'].copy()
            postprocessed = data['lsl_postprocessed_timestamps'].copy()
        sync = json.loads(archive.read(stem + '_sync_report.json'))
    cfg = decoder.Config(**{k: v for k, v in session['config'].items()
                            if k in {f.name for f in fields(decoder.Config)}})
    for name in ('channel_indices', 'channel_positions', 'window_size'):
        setattr(cfg, name, tuple(getattr(cfg, name)))
    cfg.participant_id = cfg.participant_id or session['session_id']
    cfg.calibration_file = None
    cfg.validate()
    if session['algorithm'] != 'FBCCA' or cfg.session_mode != 'cued':
        raise ValueError('本分析入口目前需要无模板 FBCCA 采集会话')
    if (cfg.notch_hz != 50.0 or cfg.target_fs <= 180
            or max(cfg.filter_bands[-1]) > cfg.target_fs / 2):
        raise ValueError('当前比较预设要求 50 Hz 工频陷波及高于180 Hz的采样率')
    if any(t['sampling_rate_hz'] != cfg.target_fs for t in trials):
        raise ValueError('本分析入口目前要求原始采样率与处理采样率相同')
    if (not np.isfinite(windows).all() or not np.isfinite(continuous).all()
            or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0)):
        raise ValueError('数据必须有限，连续时间轴必须严格递增')
    if any(len(decoder._normalised_rows(window)[1]) != len(cfg.channel_indices) for window in windows):
        raise ValueError('当前完整方案比较要求所选通道全部有效；检测到恒值通道，不能训练个人模板')
    return dict(manifest=manifest, session=session, trials=trials, windows=windows,
                saved_scores=scores, grids=grids, continuous=continuous,
                timestamps=timestamps, postprocessed_timestamps=postprocessed,
                sync=sync, cfg=cfg, constant_channels=constant_channels,
                labels=np.array([t['true_class'] for t in trials]),
                blocks=np.array([t['block_id'] for t in trials]))


def decode(window, cfg, model=None, context=None):
    return decoder.classify_eeg_window(
        window, cfg.target_fs, window_s=cfg.window_s, target_fs=cfg.target_fs,
        n_harmonics=cfg.n_harmonics, notch_hz=cfg.notch_hz,
        a=cfg.weight_a, b=cfg.weight_b, regularization=cfg.cca_regularization,
        min_valid_channels=cfg.min_valid_channels, filter_bands=cfg.filter_bands,
        line_regression_hz=cfg.line_regression_hz, calibration=model, notch_context=context)


def history_context(data, index):
    """用真实录制切片经过在线 epoch 入口，复用边界和缺口检查。"""
    cfg, stamps = data['cfg'], data['timestamps']
    grid, trial = data['grids'][index], data['trials'][index]
    left = int(np.searchsorted(stamps, grid[0], side='right') - 1)
    # 快照保留 end 之后一个支撑点；在线 epoch 仅把 <=end 的点用于陷波。
    right = int(np.searchsorted(stamps, trial['requested_window_end'], side='left') + 1)
    history = int(round(cfg.notch_history_s * cfg.target_fs))
    if left < history:
        raise ValueError('缺少足够历史原始数据')
    raw = data['continuous'][:, left-history:right]
    received = data['postprocessed_timestamps'][left-history:right]
    stream = SimpleNamespace(get_data=lambda picks: (raw[picks], received))
    reader = decoder.MNEStreamWindow(stream, range(raw.shape[0]), cfg.device_timestamp_lag_s)
    epoch = reader.epoch(trial['requested_window_start'], cfg.window_s, cfg.target_fs,
        cfg.max_gap_factor, cfg.rate_tolerance, cfg.hard_gap_factor,
        cfg.hard_rate_tolerance, cfg.max_warning_gaps, notch_history_s=cfg.notch_history_s)
    if (not np.array_equal(epoch.uniform_timestamps, grid)
            or not np.allclose(epoch.data, data['windows'][index], rtol=1e-10, atol=1e-10)):
        raise ValueError('连续数据重建与录制窗口不一致，拒绝继续评估')
    return epoch.notch_context


def accuracy(labels, scores):
    return float(np.mean(np.argmax(scores, axis=-1) + 1 == labels))


def select_candidate(validation_scores, y, candidate_names):
    """只由传入的验证集选参数；平分时固定取声明顺序靠前的方案。"""
    return max(candidate_names, key=lambda name: accuracy(y, validation_scores[name]))


def nested_group_validation(data, predictions, settings):
    """外层留整轮；模板在内层重新训练，外层测试标签不参与选择。"""
    x, y, groups = (data[k] for k in ('windows', 'labels', 'blocks'))
    names, unique = list(predictions), np.unique(groups)
    output, folds = np.empty((len(y), 40)), []
    for held in unique:
        train, test = groups != held, groups == held
        train_indices = np.flatnonzero(train)
        validation = {}
        for name in names:
            if not name.endswith('_ecca'):
                validation[name] = predictions[name][train]
                continue
            # 不能使用外层 OOF 数组作为内层验证：其中的模型会见到外层测试块。
            inner_scores = np.empty((len(train_indices), 40))
            candidate = settings[name]
            for inner_held in unique[unique != held]:
                inner_train = train & (groups != inner_held)
                inner_test = train & (groups == inner_held)
                model = decoder.PersonalCalibration.fit(
                    x[inner_train], y[inner_train], candidate.target_fs, candidate,
                    sources=[{'trial_ids': [data['trials'][i]['trial_id']
                                           for i in np.flatnonzero(inner_train)]}])
                indices = np.flatnonzero(inner_test)
                inner_scores[np.isin(train_indices, indices)] = [
                    decode(x[i], candidate, model)['scores'] for i in indices]
            validation[name] = inner_scores
        chosen = select_candidate(validation, y[train], names)
        output[test] = predictions[chosen][test]
        folds.append({'test_block': int(held), 'selected': chosen,
            'selection_trial_ids': [data['trials'][i]['trial_id'] for i in np.flatnonzero(train)],
            'test_trial_ids': [data['trials'][i]['trial_id'] for i in np.flatnonzero(test)],
            'inner_accuracy': {name: accuracy(y[train], score) for name, score in validation.items()},
            'test_accuracy': accuracy(y[test], output[test])})
        print(f'外层 block {held}: {chosen}, 准确率 {folds[-1]["test_accuracy"]:.1%}', flush=True)
    return output, folds


def result_metrics(y, scores, blocks):
    from scipy.stats import binomtest
    predicted = np.argmax(scores, axis=-1) + 1
    correct = int(np.sum(predicted == y))
    ci = binomtest(correct, len(y)).proportion_ci(method='wilson')
    q = next(t for t in decoder.TARGETS if t.symbol == 'Q')
    non_q = y != q.class_id
    false_q = int(np.sum((predicted == q.class_id) & non_q))
    return {'correct': correct, 'n': len(y), 'accuracy': correct / len(y),
        'wilson_95_descriptive': [float(ci.low), float(ci.high)],
        'by_block': {str(g): accuracy(y[blocks == g], scores[blocks == g]) for g in np.unique(blocks)},
        'prediction_counts': np.bincount(predicted - 1, minlength=40).tolist(),
        'q_bias': {'class_id': q.class_id, 'frequency_hz': q.frequency_hz,
            'true_q_trials': int(np.sum(~non_q)), 'non_q_trials': int(np.sum(non_q)),
            'predicted_q': int(np.sum(predicted == q.class_id)),
            'correct_q': int(np.sum((predicted == q.class_id) & ~non_q)),
            'false_q': false_q,
            'false_q_rate_on_non_q': false_q / int(np.sum(non_q)) if np.any(non_q) else None}}


def signal_diagnostics(data):
    """保存可复用清理窗；频谱以输入数值单位描述，不推断微伏。"""
    from scipy.signal import filtfilt, iirnotch, periodogram
    x, cfg = data['windows'], data['cfg']
    if cfg.notch_hz is None:
        raise ValueError('工频比较要求会话明确记录 notch_hz')
    bn, an = iirnotch(cfg.notch_hz, 30, fs=cfg.target_fs)
    epoch = filtfilt(bn, an, x, axis=-1)
    history = np.array([decoder.apply_history_notch(window, cfg.target_fs, cfg.notch_hz,
                       history_context(data, i))[0] for i, window in enumerate(x)])
    spectra, stats = {}, {}
    for name, values in [('raw', x), ('epoch_notch', epoch), ('history_notch', history)]:
        frequencies, psd = periodogram(values, fs=cfg.target_fs, window='hann', detrend='constant', axis=-1)
        line = (frequencies >= cfg.notch_hz - 1) & (frequencies <= cfg.notch_hz + 1)
        reference = (frequencies >= 6) & (frequencies <= 90) & (frequencies < cfg.target_fs / 2)
        fractions = psd[..., line].sum(-1) / psd[..., reference].sum(-1)
        trial_medians = np.median(fractions, axis=-1)
        stats[name] = {'median_line_fraction': float(np.median(trial_medians)),
            'min_trial_line_fraction': float(np.min(trial_medians)),
            'max_trial_line_fraction': float(np.max(trial_medians)),
            'line_fraction_by_channel': np.median(fractions, axis=0).tolist(),
            'trial_line_fractions': trial_medians.tolist()}
        spectra[name] = np.median(psd, axis=(0, 1))
    stamps, trials = data['timestamps'], data['session']['trials']
    timing = {'samples': len(stamps), 'duration_s': float(stamps[-1] - stamps[0]),
        'postprocessed_rate_hz': float((len(stamps) - 1) / (stamps[-1] - stamps[0])),
        'timestamp_gaps_over_1p5_samples': int(np.sum(np.diff(stamps) > 1.5 / cfg.target_fs)),
        'timestamp_nonpositive_intervals': int(np.sum(np.diff(stamps) <= 0)),
        'max_interval_ms': float(np.max(np.diff(stamps)) * 1000),
        'physical_alignment': data['sync'].get('physical_alignment', 'unverified')}
    qc = {'attempted': len(trials), 'valid': len(x),
        'excluded': [{'trial_id': t['trial_id'], 'status': t['status'], 'reason': t['reason']}
                     for t in trials if t['status'] != 'valid'],
        'spectrum': stats, 'timing': timing, 'constant_lsl_channels': data['constant_channels'],
        'identical_adjacent_selected_vectors': int(np.sum(np.all(np.diff(data['continuous'], axis=-1) == 0, axis=0))),
        'channel_labels': list(cfg.channel_labels), 'channel_indices': list(cfg.channel_indices),
        'electrode_positions_assumed': False,
        'declared_units': [c.get('unit', 'UNKNOWN') for c in data['manifest']['acquisition_metadata']['selected_channels']],
        'unit': 'unknown unless verified by acquisition metadata; no physical amplitude inference',
        'spectrum_method': 'Hann periodogram, 49–51 Hz / 6–90 Hz for 50 Hz notch; median across channels then trials',
        'frequency_resolution_hz': float(frequencies[1] - frequencies[0]),
        'interpretation': 'valid 是原程序有效试次标记，不等于电极质量合格；软件时间轴不能证明物理同步'}
    return qc, history, frequencies, spectra


def write_report(output, data, summary, predictions, frequencies, spectra):
    """本地自包含 HTML 和可分享 PNG；图表由 matplotlib 生成。"""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib import font_manager
    font_names = {f.name for f in font_manager.fontManager.ttflist}
    for font in ['PingFang HK', 'PingFang SC', 'Arial Unicode MS', 'Heiti TC', 'SimHei']:
        if font in font_names:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False
    cfg = data['cfg']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    names = list(predictions)
    labels = [DISPLAY_NAMES.get(n, n) for n in names]
    values = [summary['results'][n]['accuracy'] * 100 for n in names]
    bars = axes[0, 0].barh(labels, values, color=['#0d9488' if n == 'nested_selection' else '#4169a1' for n in names])
    axes[0, 0].bar_label(bars, fmt='%.1f%%', padding=3)
    axes[0, 0].set(xlim=(0, max(values) + 12), xlabel='准确率 (%)', title=f'同一会话内评估 · {len(data["labels"])} 个有效试次')
    axes[0, 0].invert_yaxis()
    for name, label in [('raw', '原始窗'), ('epoch_notch', '短窗陷波'), ('history_notch', f'{cfg.notch_history_s:g} 秒历史陷波')]:
        axes[0, 1].semilogy(frequencies, spectra[name], label=label)
    axes[0, 1].set(xlim=(6, 90), xlabel='频率 (Hz)', ylabel='PSD（输入单位²/Hz）', title='通道与试次 PSD 中位数')
    axes[0, 1].legend()
    for name, label in [('m3_epoch', 'M3 短窗对照'), ('m3_history', '历史陷波'), ('nested_selection', '嵌套选择')]:
        if name in summary['results']:
            values_by_block = summary['results'][name]['by_block']
            axes[1, 0].plot(list(values_by_block), [v * 100 for v in values_by_block.values()], 'o-', label=label)
    axes[1, 0].set(xlabel='轮次', ylabel='准确率 (%)', ylim=(0, 100), title='分轮结果（每轮 40 个目标）')
    axes[1, 0].legend()
    matrix = np.zeros((40, 40), dtype=int)
    pred = np.argmax(predictions['m3_history'], axis=1)
    np.add.at(matrix, (data['labels'] - 1, pred), 1)
    axes[1, 1].imshow(matrix, origin='upper', cmap='Blues', vmin=0, vmax=len(np.unique(data['blocks'])))
    axes[1, 1].set(xlabel='预测类别（1–40）', ylabel='真实类别（1–40）', title='历史陷波混淆矩阵')
    axes[1, 1].set_xticks([0, 9, 19, 29, 39], [1, 10, 20, 30, 40])
    axes[1, 1].set_yticks([0, 9, 19, 29, 39], [1, 10, 20, 30, 40])
    fig.savefig(output / 'analysis.png', dpi=160)
    plt.close(fig)
    qc, nested = summary['quality'], summary['results']['nested_selection']
    original = summary['recorded_online']
    block_header = ' / '.join(original['by_block'])
    n_blocks = len(np.unique(data['blocks']))
    rows = ''.join('<tr><td>' + html.escape(DISPLAY_NAMES.get(n, n)) + '</td><td>'
                   + f"{r['correct']}/{r['n']} ({r['accuracy']:.1%})" + '</td><td>'
                   + ' / '.join(f'{v:.1%}' for v in r['by_block'].values()) + '</td><td>'
                   + f"{r['q_bias']['predicted_q']} / {r['q_bias']['false_q']}" + '</td></tr>'
                   for n, r in summary['results'].items())
    folds = ''.join(f"<li>测试第 {f['test_block']} 轮：内层选择 {html.escape(DISPLAY_NAMES[f['selected']])}，"
                    f"外层准确率 {f['test_accuracy']:.1%}。</li>" for f in summary['nested_folds'])
    line_rows = ''.join(f"<tr><td>{html.escape(p)}</td><td>{v:.2%}</td></tr>"
                        for p, v in zip(qc['channel_labels'], qc['spectrum']['raw']['line_fraction_by_channel']))
    import base64
    png = base64.b64encode((output / 'analysis.png').read_bytes()).decode()
    source = html.escape(str(summary['source_file']))
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>EEG 离线分析</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;color:#183149;max-width:1100px;margin:45px auto;padding:0 24px;background:#f8fafc}}
h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border-bottom:1px solid #dce4ec;padding:9px;text-align:left}}
.lead{{padding:18px 22px;background:#e2f3f0;border-left:5px solid #0d9488}}img{{width:100%}}code{{overflow-wrap:anywhere}}</style>
<h1>EEG 离线分析与代码优化</h1><p>会话：{html.escape(data['session']['session_id'])} · {qc['valid']} 个有效试次 · 40 类 · {n_blocks} 轮</p>
<div class="lead">原录制方案 {original['accuracy']:.1%}；固定历史陷波方案 {summary['results']['m3_history']['accuracy']:.1%}；
参数选择纳入分轮嵌套验证后 {nested['accuracy']:.1%}。这些结果需要用新会话验证，未评估自由输入中的空闲误触发。</div>
<h2>数据与主要发现</h2><p>共 {qc['attempted']} 次尝试，{len(qc['excluded'])} 次无效并重试。
滤波前 49–51 Hz 能量占 6–90 Hz 能量的通道中位数为 {qc['spectrum']['raw']['median_line_fraction']:.2%}。
原录制方案 {original['prediction_counts'][10]} 次判为 Q（10 Hz）；50 Hz 恰好与其第 5 谐波重合。
若 10 Hz 预测集中，应检查工频残余是否参与打分，但不能据此确定干扰的物理来源。</p>
<p>连续记录 {qc['timing']['samples']:,} 点，{qc['timing']['duration_s']:.2f} 秒，软件时间轴估计 {qc['timing']['postprocessed_rate_hz']:.4f} Hz。
大于 1.5 个采样周期的间隔 {qc['timing']['timestamp_gaps_over_1p5_samples']} 处，非递增间隔 {qc['timing']['timestamp_nonpositive_intervals']}。
LSL 恒定通道索引为 {qc['constant_lsl_channels']}，当前解码选择 {qc['channel_indices']}。
相邻所选通道向量完全重复 {qc['identical_adjacent_selected_vectors']} 次；这些统计不能证明硬件没有丢样，也不能确认设备原始采样率。</p>
<p>无效试次原因：{html.escape('; '.join(t['reason'] for t in qc['excluded']) or '无')}。</p>
<h2>算法比较</h2><table><tr><th>方案</th><th>总正确率</th><th>第 {block_header} 轮</th><th>预测 Q / 其中错判 Q</th></tr>{rows}</table>
<p>真实 Q 试次 {original['q_bias']['true_q_trials']} 个，非 Q 试次 {original['q_bias']['non_q_trials']} 个。
原录制方案错判 Q {original['q_bias']['false_q']} 次；历史陷波错判 Q {summary['results']['m3_history']['q_bias']['false_q']} 次。
减少预测 Q 的次数本身不等于提升识别能力，需同时检查真实 Q 的命中数与整体准确率；本表保留所有40个目标。</p>
<p>所有方案使用相同的 {cfg.window_s:g} 秒、{len(cfg.channel_indices)} 通道窗，响应起点保留 {cfg.response_delay_s * 1000:g} ms；40 个类别全部参与分类。
未根据标签删除试次或通道，未使用未来真实样本做滤波。历史陷波使用窗口前 {cfg.notch_history_s:g} 秒原始数据，按原始采样率陷波后插值回原分析网格。
这同时改变陷波的位置及左边界上下文，因此不能把全部提升仅归因于更长历史。</p>
<h2>防止训练/测试混用</h2><p>原线上分数最大重放误差为 {summary['replay_max_score_error']:.3g}。
eCCA 每次只用其他轮训练，再测试完整留出轮；未载入 ZIP 中全量训练的 PKL。
外层留一整轮，内层在其余轮次做留轮训练/验证，用固定声明顺序处理平分；外层测试标签不参与方案选择。</p>
<ul>{folds}</ul><p>嵌套结果的描述性 Wilson 95% 区间为 {nested['wilson_95_descriptive'][0]:.1%}–{nested['wilson_95_descriptive'][1]:.1%}，
该区间按试次二项分布计算，未建模轮间相关性。候选方案来自本场问题诊断，这仍是同一受试者、同一会话内的探索性验证，不代表跨日或新会话性能。
本工具不自动部署个人模板。全数据挑选出的最佳值应视为描述性结果。</p>
<h2>频谱与通道</h2><img alt="识别准确率、PSD、分轮结果及混淆矩阵" src="data:image/png;base64,{png}">
<table><tr><th>接收通道编号（不指定头皮位置）</th><th>滤波前工频占比中位数</th></tr>{line_rows}</table>
<p>短窗陷波后工频占比中位数 {qc['spectrum']['epoch_notch']['median_line_fraction']:.2%}，历史陷波后为 {qc['spectrum']['history_notch']['median_line_fraction']:.2%}。
LSL 记录单位为 {html.escape(str(qc['declared_units']))}；图表使用输入数值单位，未换算微伏幅值。valid 仅表示原程序质量门控通过，不能等同于电极信号合格。</p>
<h2>代码与下一次采集</h2><p>已增加无 UI/LSL 的离线分析入口、逐试次 CSV、候选分数 NPZ、分轮验证来源记录、清理后 EEG 与本报告。
项目附 history 复测启动器，使用 2 秒窗、M3 频带、140 ms 起点及新的目标顺序，这是针对 20261001 会话得到的固定复测方案；原启动器继续支持既有参数和个人模板。</p>
<p>优先在 OpenBCI GUI 核对参考/地电极接触、各电极接触和附近工频干扰，再采集一场新的完整提示测试。
先比较滤波前工频占比，再评估识别准确率。时间校准需要光电/触发或其他独立证据；不能用最高分类准确率推断设备物理延迟。</p>
<h2>复现与文件</h2><p>输入：<code>{source}</code><br>SHA-256：<code>{summary['source_sha256']}</code></p>
<p><code>FBCCA_NO_PAUSE=1 ./start_keyboard_dual_mode.command --offline-session session_records/{html.escape(Path(summary['source_file']).name)} --offline-output offline_results/{html.escape(data['session']['session_id'])}</code></p>
<p>analysis_results.json：完整统计与验证来源；trial_predictions.csv：逐试次真实/预测标签；candidate_scores.npz：完整分数；
processed_eeg.npz：原始均匀窗、历史陷波窗及去直流版本（未归一化）、时间轴、标签与元数据；analysis.png：图表。</p>
<h2>方法参考</h2><p><a href="https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140703">Nakanishi 等，2015：CCA 与个人模板比较</a>；
<a href="https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.signal.filtfilt.html">SciPy filtfilt 文档：边界瞬态与滤波行为</a>。
准确率、工频与时间轴结论均来自本地数据复算；处理方案由现有在线函数执行。</p></html>'''
    (output / 'report.html').write_text(report, encoding='utf-8')


DISPLAY_NAMES = {
    'm3_epoch': 'M3 / 短窗陷波对照', 'm3_epoch_ecca': 'M3 短窗 + 个人模板',
    'm3_regression50': 'M3 + 50 Hz 回归', 'm3_regression50_ecca': 'M3 + 回归 + 个人模板',
    'line_robust_epoch': '48 Hz / 3 谐波', 'line_robust_epoch_ecca': '48 Hz + 个人模板',
    'line_robust_regression50': '48 Hz + 50 Hz 回归',
    'line_robust_regression50_ecca': '48 Hz + 回归 + 个人模板',
    'm3_history': 'M3 / 历史陷波', 'nested_selection': '分轮嵌套选择（主评估）',
}


def run(path, output):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_session(path)
    cfg, x, y, groups = (data[k] for k in ('cfg', 'windows', 'labels', 'blocks'))
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError('至少需要 3 个完整 block，才能进行分组嵌套验证')
    for group in unique:
        if not np.array_equal(np.sort(y[groups == group]), np.arange(1, 41)):
            raise ValueError(f'block {group} 需要每类恰好一个有效试次，不能进行完整模板验证')
    epoch_cfg = replace(cfg, notch_mode='epoch', line_regression_hz=None)
    candidates = [
        ('m3_epoch', decoder.configure_fbcca_profile(epoch_cfg, 'm3')),
        ('m3_regression50', replace(decoder.configure_fbcca_profile(epoch_cfg, 'm3'), line_regression_hz=50.0)),
        ('line_robust_epoch', decoder.configure_fbcca_profile(epoch_cfg, 'line_robust')),
        ('line_robust_regression50', replace(decoder.configure_fbcca_profile(epoch_cfg, 'line_robust'), line_regression_hz=50.0)),
        ('m3_history', replace(decoder.configure_fbcca_profile(epoch_cfg, 'm3'), notch_mode='history')),
    ]
    predictions, settings = {}, {}
    recorded_scores = np.array([decode(window, cfg, context=history_context(data, i)
                               if cfg.notch_mode == 'history' else None)['scores'] for i, window in enumerate(x)])
    replay_error = float(np.max(np.abs(recorded_scores - data['saved_scores'])))
    if not np.allclose(recorded_scores, data['saved_scores'], rtol=1e-6, atol=1e-8):
        raise ValueError(f'无法重现在线分数，最大误差 {replay_error}')
    for name, candidate in candidates:
        candidate.validate()
        results = [decode(window, candidate, context=history_context(data, i)
                          if candidate.notch_mode == 'history' else None)
                   for i, window in enumerate(x)]
        scores = np.array([r['scores'] for r in results])
        predictions[name] = scores
        settings[name] = candidate
        print(name, accuracy(y, scores), flush=True)
        if candidate.notch_mode == 'history':
            continue
        loo = np.empty((len(y), 40))
        for held in unique:
            train, test = groups != held, groups == held
            model = decoder.PersonalCalibration.fit(x[train], y[train], cfg.target_fs, candidate)
            loo[test] = np.array([decode(window, candidate, model)['scores'] for window in x[test]])
        predictions[name + '_ecca'] = loo
        settings[name + '_ecca'] = candidate
        print(name + '_ecca', accuracy(y, loo), flush=True)
    nested_scores, folds = nested_group_validation(data, predictions, settings)
    chosen = select_candidate(predictions, y, list(predictions))
    predictions['nested_selection'] = nested_scores
    quality, history, frequencies, spectra = signal_diagnostics(data)
    summary = {'schema_version': 1, 'source_file': str(Path(path).resolve()),
               'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
               'replay_max_score_error': replay_error,
               'recorded_online': result_metrics(y, recorded_scores, groups),
               'quality': quality, 'nested_folds': folds,
               'candidate_order': list(settings), 'candidate_configs': {name: asdict(c) for name, c in settings.items()},
               'tie_break': 'first in candidate_order',
               'selected_for_prospective_retest': chosen,
               'validation_scope': 'exploratory same-session nested leave-one-block-out; new session required',
               'results': {name: result_metrics(y, scores, groups) for name, scores in predictions.items()},
               'decoder_source_sha256': hashlib.sha256(Path(decoder.__file__).read_bytes()).hexdigest(),
               'analysis_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                            'scipy': version('scipy'), 'matplotlib': version('matplotlib')}}
    summary['elapsed_s'] = time.perf_counter() - started
    (output / 'analysis_results.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    np.savez_compressed(output / 'candidate_scores.npz', labels=y, blocks=groups,
                        trial_ids=[t['trial_id'] for t in data['trials']], **predictions)
    np.savez_compressed(output / 'processed_eeg.npz', raw_uniform_eeg=x,
        history_notched_eeg=history, history_notched_demeaned_eeg=history-history.mean(-1, keepdims=True),
        uniform_lsl_timestamps=data['grids'], labels=y, blocks=groups,
        trial_ids=[t['trial_id'] for t in data['trials']], channel_labels=cfg.channel_labels,
        channel_indices=cfg.channel_indices,
        sampling_rate_hz=cfg.target_fs,
        metadata_json=json.dumps({'source_sha256': summary['source_sha256'],
            'source_file': summary['source_file'], 'axis_order': 'trial, channel, sample',
            'processing': 'native raw history notch Q=30; interpolate to original epoch grid; no future samples',
            'history_s': cfg.notch_history_s, 'notch_hz': cfg.notch_hz,
            'response_delay_s': cfg.response_delay_s, 'units': quality['unit']}, ensure_ascii=False))
    with (output / 'trial_predictions.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['trial_id', 'block_id', 'true_class', 'symbol', 'frequency_hz', *predictions])
        for i, trial in enumerate(data['trials']):
            writer.writerow([trial['trial_id'], trial['block_id'], int(y[i]), trial['true_symbol'],
                             trial['true_frequency_hz'], *[int(np.argmax(s[i])) + 1 for s in predictions.values()]])
    write_report(output, data, summary, predictions, frequencies, spectra)
    print(f'报告已保存：{(output / "report.html").resolve()}', flush=True)
    return data, predictions, settings, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session_zip', type=Path)
    parser.add_argument('--output', type=Path, default=Path('offline_results'))
    args = parser.parse_args()
    run(args.session_zip, args.output)
