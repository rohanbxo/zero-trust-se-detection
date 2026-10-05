"""Generate numbers.tex (LaTeX macros) from results_v3.json.

Usage:  python gen_numbers.py results_v3.json  > numbers.tex
        python gen_numbers.py                  > numbers.tex   (placeholders only)

Every reportable number in main.tex is a macro defined here, so the paper can never
contain two different values for the same quantity. Missing values render as a red ??.
"""
import json, sys

V = {'SecureBERT only': 'SB', 'ModernBERT only': 'MB', 'Simple average (no training)': 'Avg',
     'MLP fusion, no conflict': 'NoConf', 'Proposed (MLP + conflict)': 'Full'}
FULL = 'Proposed (MLP + conflict)'
P = {'clean': 'Clean', 'typo_10': 'Typo', 'leet_obfuscation': 'Leet',
     'spaced_obfuscation': 'Spaced', 'early_50pct_turns': 'Early', 'paraphrase': 'Para'}
SEV = {'SecureBERT': 'SB', 'ModernBERT': 'MB', 'Conflict |p_SB - p_MB|': 'Conf',
       'SecureBERT (p_SB)': 'SB', 'ModernBERT (p_MB)': 'MB'}

R = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
M = {}


def put(name, fn):
    try:
        v = fn()
        M[name] = None if v is None else str(v)
    except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
        M[name] = None


def d3(x): return f'{x:.3f}'
def d4(x): return f'{x:.4f}'
def d2(x): return f'{x:.2f}'
def d1(x): return f'{x:.1f}'
def pc(x): return f'{x * 100:.1f}\\%'
def pv(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'
def sgn(x): return ('$+$' if x >= 0 else '$-$') + f'{abs(x):.3f}'


# ---------------- Data -------------------------------------------------------
put('NTrain', lambda: R['data']['n_train'])
put('NVal', lambda: R['data']['n_val'])
put('NTest', lambda: R['data']['n_test'])
put('NSevsimTest', lambda: R['data']['n_sevsim_test'])
put('SBTruncPct', lambda: d1(R['token_lengths']['train']['sb_pct_truncated_512']) + '\\%')
put('MBTokMeanTrain', lambda: f"{R['token_lengths']['train']['mb_mean']:,.0f}")
put('MBTokMeanTest', lambda: f"{R['token_lengths']['test']['mb_mean']:,.0f}")
put('MBTokMaxTest', lambda: f"{R['token_lengths']['test']['mb_max']:,}")
put('MBTokMinTest', lambda: f"{R['token_lengths']['test']['mb_min']:,}")

# ---------------- Frozen config / calibration (Protocol A) -------------------
fc = lambda: R['frozen_config']
put('TauSB', lambda: d4(fc()['T_securebert']))
put('TauMB', lambda: d4(fc()['T_modernbert']))
put('Tblock', lambda: d1(fc()['T_block']))
put('Rblock', lambda: d3(fc()['R_block']))
put('Tpermit', lambda: d1(fc()['T_permit']))
put('YoudenJ', lambda: d4(fc()['youden_J_val']))
for enc, tag in [('securebert', 'SB'), ('modernbert', 'MB')]:
    for split in ['val', 'test']:
        for m in ['brier', 'ece']:
            for st in ['raw', 'cal']:
                put(f'{tag}{m.capitalize()}{split.capitalize()}{st.capitalize()}',
                    lambda enc=enc, split=split, m=m, st=st: d4(R['calibration'][enc][f'{split}_{m}_{st}']))
put('MLPEpochs', lambda: R['mlp_cv_on_validation']['epochs'])
put('ValCVFone', lambda: d3(R['mlp_cv_on_validation']['full']['f1_mean']))
put('ValCVFoneSD', lambda: d3(R['mlp_cv_on_validation']['full']['f1_std']))

# ---------------- Protocol A: strict transfer, SEConvo test ------------------
tm = lambda: R['test_main']
for k, f in [('f1', d3), ('accuracy', pc), ('recall', pc), ('precision', pc), ('auroc', d3), ('fpr', pc)]:
    put('A' + {'f1': 'Fone', 'accuracy': 'Acc', 'recall': 'Rec', 'precision': 'Prec',
               'auroc': 'AUROC', 'fpr': 'FPR'}[k], lambda k=k, f=f: f(tm()[k]))
put('AFoneLo', lambda: d3(tm()['ci95_f1'][0]))
put('AFoneHi', lambda: d3(tm()['ci95_f1'][1]))
for k in ['TP', 'FP', 'FN', 'TN']:
    put('A' + k, lambda k=k: tm()[k])
put('ASEBlocked', lambda: tm()['se_blocked'])
put('ASEBlockedPct', lambda: pc(tm()['se_blocked'] / tm()['n_se']))
put('ASEFull', lambda: tm()['se_full_access'])
put('NSE', lambda: tm()['n_se'])
put('ABenFull', lambda: tm()['zones'].get('Full Access', 0) - tm()['se_full_access'])
put('NBenign', lambda: R['data']['n_test'] - tm()['n_se'])
put('SVZoneFull', lambda: R['sevsim_external']['zones']['Full Access'])
put('SVZoneBlock', lambda: R['sevsim_external']['zones']['Blocked'])
put('SVZoneRestr', lambda: R['sevsim_external']['zones']['Restricted'])
put('SVOracle', lambda: d3(R['sevsim_external']['oracle_threshold_f1_reference_only']))
put('AAUROCLo', lambda: d3(tm()['ci95_auroc'][0]))
put('AAUROCHi', lambda: d3(tm()['ci95_auroc'][1]))
put('LatCorr', lambda: d2(R['latency']['corr_with_length']))
put('SBTruncPctTest', lambda: d1(R['token_lengths']['test']['sb_pct_truncated_512']) + '\\%')
for z, tag in [('Full Access', 'Full'), ('Restricted', 'Restr'), ('Blocked', 'Block')]:
    put(f'AZone{tag}', lambda z=z: tm()['zones'].get(z, 0))
    put(f'AZone{tag}Pct', lambda z=z: pc(tm()['zones'].get(z, 0) / R['data']['n_test']))

for v, t in V.items():
    a = lambda v=v: R['ablation_test'][v]
    put(f'A{t}Fone', lambda a=a: d3(a()['f1']))
    put(f'A{t}Acc', lambda a=a: pc(a()['accuracy']))
    put(f'A{t}Rec', lambda a=a: pc(a()['recall']))
    put(f'A{t}Prec', lambda a=a: pc(a()['precision']))
    put(f'A{t}AUROC', lambda a=a: d3(a()['auroc']))
    put(f'A{t}FPR', lambda a=a: pc(a()['fpr']))
    if v != FULL:
        s = lambda v=v: R['significance_vs_full'][v]
        put(f'A{t}b', lambda s=s: s()['b'])
        put(f'A{t}c', lambda s=s: s()['c'])
        put(f'A{t}p', lambda s=s: pv(s()['pvalue']))
        put(f'A{t}pHolm', lambda s=s: pv(s()['p_holm']))
        put(f'A{t}dAUCLo', lambda s=s: sgn(s()['delta_auroc_ci95'][0]))
        put(f'A{t}dAUCHi', lambda s=s: sgn(s()['delta_auroc_ci95'][1]))

# ---------------- Protocol B: cross-fitted SEConvo ---------------------------
cf = lambda: R['crossfit_seconvo']
for v, t in V.items():
    sm = lambda v=v: cf()['summary'][v]
    put(f'B{t}Fone', lambda sm=sm: d3(sm()['f1'][0]))
    put(f'B{t}FoneSD', lambda sm=sm: d3(sm()['f1'][1]))
    put(f'B{t}AUROC', lambda sm=sm: d3(sm()['auroc'][0]))
    put(f'B{t}AUROCSD', lambda sm=sm: d3(sm()['auroc'][1]))
    put(f'B{t}Rec', lambda sm=sm: pc(sm()['recall'][0]))
    put(f'B{t}Prec', lambda sm=sm: pc(sm()['precision'][0]))
    put(f'B{t}Acc', lambda sm=sm: pc(sm()['accuracy'][0]))
s0 = lambda: cf()['seed0']
for k, f in [('f1', d3), ('accuracy', pc), ('recall', pc), ('precision', pc), ('auroc', d3), ('fpr', pc)]:
    put('Bz' + {'f1': 'Fone', 'accuracy': 'Acc', 'recall': 'Rec', 'precision': 'Prec',
                'auroc': 'AUROC', 'fpr': 'FPR'}[k], lambda k=k, f=f: f(s0()[k]))
put('BzFoneLo', lambda: d3(s0()['ci95_f1'][0]))
put('BzFoneHi', lambda: d3(s0()['ci95_f1'][1]))
for k in ['TP', 'FP', 'FN', 'TN']:
    put('Bz' + k, lambda k=k: s0()[k])
for z, tag in [('Full Access', 'Full'), ('Restricted', 'Restr'), ('Blocked', 'Block')]:
    put(f'BZone{tag}', lambda z=z: s0()['zones'].get(z, 0))
put('BTblockMin', lambda: d1(min(d['T_block'] for d in s0()['thresholds_per_fold'])))
put('BTblockMax', lambda: d1(max(d['T_block'] for d in s0()['thresholds_per_fold'])))
put('BTpermitMin', lambda: d1(min(d['T_permit'] for d in s0()['thresholds_per_fold'])))
put('BTpermitMax', lambda: d1(max(d['T_permit'] for d in s0()['thresholds_per_fold'])))
put('BTauSBMin', lambda: d2(min(d['T_sb'] for d in s0()['thresholds_per_fold'])))
put('BTauSBMax', lambda: d2(max(d['T_sb'] for d in s0()['thresholds_per_fold'])))
put('BTauMBMin', lambda: d2(min(d['T_mb'] for d in s0()['thresholds_per_fold'])))
put('BTauMBMax', lambda: d2(max(d['T_mb'] for d in s0()['thresholds_per_fold'])))
for v, t in list(V.items()) + [('Mistral-7B (thr 50)', 'Mistral')]:
    if v == FULL:
        continue
    m = lambda v=v: s0()['mcnemar'][v]
    put(f'B{t}b', lambda m=m: m()['b'])
    put(f'B{t}c', lambda m=m: m()['c'])
    put(f'B{t}p', lambda m=m: pv(m()['pvalue']))
    put(f'B{t}pHolm', lambda m=m: pv(m()['p_holm']))

# ---------------- Mistral-7B baseline ----------------------------------------
mi = lambda: R['mistral7b']['a_priori_threshold_50']
put('MisFone', lambda: d3(mi()['f1']))
put('MisAcc', lambda: pc(mi()['accuracy']))
put('MisRec', lambda: pc(mi()['recall']))
put('MisPrec', lambda: pc(mi()['precision']))
def _mis_fpr():
    m = mi(); n_se, n_ben = 191, 169
    tp = round(m['recall'] * n_se); fp = round(tp / m['precision']) - tp
    return pc(fp / n_ben)
put('MisFPR', _mis_fpr)
put('MisAUROC', lambda: d3(R['mistral7b']['auroc_all_360_neutral_imputed']))

# ---------------- SE-VSim held-out test --------------------------------------
for v, t in V.items():
    sv = lambda v=v: R['sevsim_external']['per_variant'][v]
    put(f'SV{t}Fone', lambda sv=sv: d3(sv()['f1']))
    put(f'SV{t}AUROC', lambda sv=sv: d3(sv()['auroc']))
    put(f'SV{t}Rec', lambda sv=sv: pc(sv()['recall']))
    put(f'SV{t}Prec', lambda sv=sv: pc(sv()['precision']))
    put(f'SV{t}Acc', lambda sv=sv: pc(sv()['accuracy']))

# ---------------- Robustness (cross-fit protocol) -----------------------------
rc = lambda: R['robustness_crossfit']
for p, pt in P.items():
    for v, t in V.items():
        put(f'Rob{pt}{t}AUROC', lambda p=p, v=v: d3(rc()[p][v]['auroc'][0]))
        put(f'Rob{pt}{t}Fone', lambda p=p, v=v: d3(rc()[p][v]['f1'][0]))
        if p != 'clean':
            put(f'Rob{pt}{t}dAUROC', lambda p=p, v=v: sgn(rc()[p][v]['auroc'][0] - rc()['clean'][v]['auroc'][0]))
    put(f'Rob{pt}FPR', lambda p=p: pc(rc()[p][FULL]['fpr'][0]))
    put(f'Rob{pt}SEFull', lambda p=p: d1(rc()[p]['zones_proposed']['se_full_access']))
    put(f'Rob{pt}BenFull', lambda p=p: d1(rc()[p]['zones_proposed']['benign_full_access']))
    put(f'Rob{pt}SEBlocked', lambda p=p: d1(rc()[p]['zones_proposed']['se_blocked_pct']) + '\\%')

# ---------------- SHAP ---------------------------------------------------------
for key, lab in [('SecureBERT (p_SB)', 'SB'), ('ModernBERT (p_MB)', 'MB'), ('Conflict |p_SB - p_MB|', 'Conf')]:
    put(f'Shap{lab}', lambda key=key: d1(R['shap']['bootstrap_mean_pct'][key]) + '\\%')
    put(f'Shap{lab}SD', lambda key=key: d1(R['shap']['bootstrap_std_pct'][key]) + '\\%')

# ---------------- Compute & latency -------------------------------------------
put('LatMean', lambda: d1(R['latency']['mean_ms']))
put('LatMedian', lambda: d1(R['latency']['median_ms']))
put('LatSD', lambda: d1(R['latency']['std_ms']))
put('LatPninefive', lambda: d1(R['latency']['p95_ms']))
put('LatMin', lambda: d1(R['latency']['min_ms']))
put('LatMax', lambda: d1(R['latency']['max_ms']))
put('LatN', lambda: R['latency']['n'])
put('PeakGPU', lambda: f"{R['latency']['peak_gpu_mb']:,.0f}")

# ---------------- LoveFraud02 (optional) ---------------------------------------
lfs = lambda: R['lovefraud02_summary']
for k, nm in [('n','LFN'),('recall_k','LFRecK'),('median_first','LFMedFirst'),('before_pay_k','LFPayK'),('before_pay_n','LFPayN'),
              ('by10','LFByTen'),('by20','LFByTwenty'),('by50','LFByFifty'),('by100','LFByHundred'),('zone_block','LFZoneBlock'),
              ('zone_restr','LFZoneRestr'),('mb_before_pay_k','LFMBPayK'),('median_first_pay_cue','LFMedPayCue'),('median_len','LFMedLen')]:
    put(nm, lambda k=k: lfs()[k])
put('LFMeanFirst', lambda: d1(lfs()['mean_first']))
put('LFRecPct', lambda: d1(100 * lfs()['recall_k'] / lfs()['n']) + '\\%')
put('LFRecLo', lambda: d1(lfs()['recall_lo']) + '\\%')
put('LFRecHi', lambda: d1(lfs()['recall_hi']) + '\\%')
put('LFPayPct', lambda: d1(100 * lfs()['before_pay_k'] / lfs()['before_pay_n']) + '\\%')
put('LFPayLo', lambda: d1(lfs()['before_pay_lo']) + '\\%')
put('LFPayHi', lambda: d1(lfs()['before_pay_hi']) + '\\%')
lf = lambda: R['lovefraud02']['B: SEConvo-calibrated'][FULL]
put('LFRecallOnline', lambda: pc(lf()['recall_online']))
put('LFRecallFull', lambda: pc(lf()['recall_full']))
put('LFMedianTurn', lambda: f"{lf()['median_first_flag']:.0f}")
put('LFBeforePay', lambda: pc(lf()['before_payment']))

# ---------------- Threshold sensitivity (Protocol A) ---------------------------
PCT = {70: 'Seventy', 75: 'SeventyFive', 80: 'Eighty', 90: 'Ninety'}
for row_i, (pct, w) in enumerate(PCT.items()):
    r = lambda pct=pct: next(x for x in R['threshold_sensitivity']['permit'] if x['percentile'] == pct)
    put(f'SensP{w}Tpermit', lambda r=r: d1(r()['T_permit']))
    put(f'SensP{w}Full', lambda r=r: d1(r()['full_pct']) + '\\%')
    put(f'SensP{w}Restr', lambda r=r: d1(r()['restricted_pct']) + '\\%')
    put(f'SensP{w}Block', lambda r=r: d1(r()['blocked_pct']) + '\\%')
    put(f'SensP{w}SEFull', lambda r=r: r()['se_full_access'])
SH = {-0.10: 'MinusTen', -0.05: 'MinusFive', 0.0: 'Zero', 0.05: 'PlusFive', 0.10: 'PlusTen'}
for dlt, w in SH.items():
    r = lambda dlt=dlt: next(x for x in R['threshold_sensitivity']['block'] if abs(x['delta'] - dlt) < 1e-9)
    put(f'SensB{w}Tblock', lambda r=r: d1(r()['T_block']))
    put(f'SensB{w}Rec', lambda r=r: pc(r()['recall']))
    put(f'SensB{w}FPR', lambda r=r: pc(r()['fpr']))
    put(f'SensB{w}Fone', lambda r=r: d3(r()['f1']))

# ---------------- Conflict-signal diagnostic (validation) ----------------------
put('ConfLegacyAUROC', lambda: d3(R['conflict_diagnostic_val']['legacy CLS-cosine conflict']['auroc_alone']))
put('ConfProbAUROC', lambda: d3(R['conflict_diagnostic_val']['|p_SB - p_MB| conflict']['auroc_alone']))

# ---------------- Length-aware thresholds (Sections 28b/28c) -------------------
las = lambda: R['length_aware_summary']
def _la_rows():
    out = []
    for name, se, ben, bp, med, by20 in las()['rows']:
        out.append(f"{name} & {se:.1f}\\% & {ben:.1f}\\% & {bp:.1f}\\% & {med} & {by20} \\\\")
    return '\n'.join(out)
put('LATableRows', _la_rows)
put('LAYoudenTone', lambda: d1(las()['thresholds_T']['youden'][0]))
for key, nm in [('rec_g','RecG'),('rec_a','RecA'),('fpr_g','FprG'),('fpr_a','FprA')]:
    put(f'LAEarly{nm}', lambda key=key: d1(las()['per_bucket_youden']['1-5'][key]) + '\\%')
for idx, nm in enumerate(['Glob','Youd','Five','Two','One']):
    put(f'LA{nm}SE', lambda idx=idx: d1(las()['rows'][idx][1]) + '\\%')
    put(f'LA{nm}Ben', lambda idx=idx: d1(las()['rows'][idx][2]) + '\\%')
    put(f'LA{nm}Pay', lambda idx=idx: d1(las()['rows'][idx][3]) + '\\%')

# ---------------- Emit ---------------------------------------------------------
print('% AUTO-GENERATED by gen_numbers.py from results_v3.json -- do not edit by hand.')
print('\\providecommand{\\PH}{\\textcolor{red}{\\textbf{??}}}')
missing = 0
for k, v in M.items():
    if v is None:
        missing += 1
    print(f'\\newcommand{{\\{k}}}{{{v if v is not None else chr(92) + "PH"}}}')
print(f'% {len(M)} macros, {missing} missing', file=sys.stderr)
