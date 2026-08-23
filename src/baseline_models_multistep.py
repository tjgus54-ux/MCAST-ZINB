"""
SECTION BL — Baseline Models  (체크포인트 재개 版 v3)
================================================================================
MCAST-ZINB_0526 비교실험용 7개 baseline — Colab 24h 런타임 제한 대응.

[v2 → v3 핵심 수정]
  ① 출력층: F.relu → F.softplus
      relu dead neuron: pre-activation<0 이면 gradient=0 → 전부 0 수렴
      softplus = log(1+exp(x)) ≥ 0 & gradient=sigmoid(x)>0 항상 존재

  ② 손실함수: plain MSE → Weighted MSE (BL_POS_WEIGHT=20)
      zero_rate=99.56% → plain MSE는 "전부 0" 예측이 local minimum
      pos_weight=20: 비제로 샘플 gradient 20배 증폭

  ③ GWN: Adaptive Adjacency 추가 (nodevec1/2 ∈ R^{N×10})
      A_adapt = softmax(relu(E1 @ E2^T)) — Wu et al. (2019) 원논문 완전 재현
      adj_phys + adj_adapt 두 GCN 합산

  ④ ASTGCN: Spatial Attention 추가 (SpatialAttention, d_attn=32)
      원논문 V∈R^{N×N} 대신 low-rank d_attn=32 projection
      (N=5,661 규모 적용을 위한 low-rank 근사; 논문에 명시 권장)
      Flash Attention: GPU PyTorch≥2.0 시 N×N 행렬 미생성 → 메모리 효율
      CPU: N×N 행렬 생성 (128MB per batch) → GPU 강력 권장

[사용 흐름]
  1. Google Drive 마운트
  2. exec(open('/content/drive/.../baseline_models.py').read())
  3. BL_CKPT_DIR / BL_RESULTS_PATH 를 Drive 경로로 변경
  4. print_progress() 로 현황 확인
  5. 모델별 셀 독립 실행

[필요 노트북 변수]
  pivot_df, AP, train_loader, val_loader, test_loader
  device, HIDDEN_DIM, EPOCHS, LR, H_RECENT
  TRAIN_START/END, TEST_START/END

[재실행 주의]
  기존 bl_results.json의 LSTM/GRU/STGCN 결과는 zero collapse 문제로 무효.
  HA 결과만 유지하고 재실행:
    import json
    with open(BL_RESULTS_PATH) as f: r = json.load(f)
    r = {k: v for k, v in r.items() if k == 'HA'}
    with open(BL_RESULTS_PATH, 'w') as f: json.dump(r, f)
================================================================================
"""

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ══════════════════════════════════════════════════════════════════════════════
#  설정  (노트북 셀에서 직접 덮어쓰기 가능)
# ══════════════════════════════════════════════════════════════════════════════
BL_BATCH_SIZE_GNN = 8     # STGCN/DCRNN/GWN 전용 — OOM 시 4→2→1
BL_SAVE_EVERY     = 5     # 체크포인트 저장 주기 (에포크)
BL_POS_WEIGHT     = 20    # Weighted MSE 양성 클래스 가중치
H_HORIZON         = 24    # [멀티스텝] 예측 지평: 한 번에 미래 몇 시간을 예측할지
                          #   본 노트북 CSTDataset 의 H_HORIZON 과 반드시 동일해야 함

# !! 반드시 Google Drive 경로로 변경 (런타임 재시작 시 /content/ 초기화됨)
BL_CKPT_DIR       = '.'
BL_RESULTS_PATH   = 'bl_results.json'

_MODEL_ORDER = ['HA', 'LSTM', 'GRU', 'STGCN', 'DCRNN', 'GraphWaveNet', 'ASTGCN']


# ══════════════════════════════════════════════════════════════════════════════
#  BL-0A  공통 평가 유틸
# ══════════════════════════════════════════════════════════════════════════════

def _compute_metrics(all_y: np.ndarray, all_pred: np.ndarray) -> dict:
    """
    기존 evaluate() 와 완전히 동일한 지표 계산.

    [멀티스텝 대응]
      입력이 (T, N) 이든 (T, N, H) 이든 동작한다.
      (T, N, H) 인 경우:
        - MAE/RMSE/KL/분류 지표는 전체를 flatten 해 계산 (모든 horizon 합산).
        - HR@20 은 '시점별 노드 랭킹' 이므로 (시점, 노드) 축이 필요 →
          H 를 시점 축으로 펼쳐 (T*H, N) 으로 만든 뒤 원래 정의대로 계산.
    """
    all_pred = np.maximum(all_pred, 0.0)

    # ── HR@20 계산용 2D 뷰 (관측단위, N) 준비 ──
    if all_y.ndim == 3:                                  # (T, N, H)
        T, N, H = all_y.shape
        y_2d = np.transpose(all_y,    (0, 2, 1)).reshape(T * H, N)   # (T*H, N)
        p_2d = np.transpose(all_pred, (0, 2, 1)).reshape(T * H, N)
    else:                                                # (T, N)
        T, N = all_y.shape
        y_2d = all_y
        p_2d = all_pred

    y_flat    = all_y.flatten()
    pred_flat = all_pred.flatten()

    # 분포 진단
    zr_true = float(np.mean(y_flat == 0))
    zr_pred = float(np.mean(pred_flat < 0.5))
    ys, ps  = y_flat.sum(), pred_flat.sum()
    if ys > 0 and ps > 0:
        p  = y_flat    / ys + 1e-10
        q  = pred_flat / ps + 1e-10
        kl = float(np.sum(p * np.log(p / q)))
    else:
        kl = float('nan')

    # 회귀
    mae  = float(mean_absolute_error(y_flat, pred_flat))
    rmse = float(np.sqrt(mean_squared_error(y_flat, pred_flat)))
    pos  = y_flat > 0
    mae_p  = float(mean_absolute_error(y_flat[pos], pred_flat[pos])) if pos.sum() > 0 else float('nan')
    rmse_p = float(np.sqrt(mean_squared_error(y_flat[pos], pred_flat[pos]))) if pos.sum() > 0 else float('nan')

    # 분류 (threshold=0.5)
    tb = (y_flat > 0).astype(int)
    pb = (pred_flat > 0.5).astype(int)
    tp = int(((pb==1)&(tb==1)).sum())
    fp = int(((pb==1)&(tb==0)).sum())
    fn = int(((pb==0)&(tb==1)).sum())
    rec  = tp / (tp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)

    # HR@20  (관측단위별 상위 20% 노드 랭킹)
    n_slice = y_2d.shape[0]
    k = max(1, int(math.floor(0.2 * N)))
    hits = events = 0
    for t in range(n_slice):
        tn = np.where(y_2d[t] > 0)[0]
        if tn.size == 0:
            continue
        events += tn.size
        hits   += int(np.intersect1d(np.argsort(p_2d[t])[::-1][:k], tn).size)
    hr = hits / events if events > 0 else float('nan')

    return {
        'zero_rate_true': zr_true, 'zero_rate_pred': zr_pred, 'kl_div': kl,
        'mae': mae, 'rmse': rmse, 'mae_pos': mae_p, 'rmse_pos': rmse_p,
        'recall_pos': float(rec), 'precision_pos': float(prec),
        'f1_pos': float(f1), 'hr_at_20': hr,
    }


def _compute_metrics_per_horizon(all_y: np.ndarray, all_pred: np.ndarray) -> list:
    """
    [멀티스텝] horizon(1..H)별로 _compute_metrics 를 각각 계산.
    입력: (T, N, H).  반환: [{'h':1, ...지표...}, {'h':2, ...}, ...]
    각 h 를 고정하면 (T, N) 슬라이스가 되어 단일스텝과 동일한 정의로 계산된다.
    """
    assert all_y.ndim == 3, f'per-horizon 은 (T,N,H) 입력 필요, got {all_y.shape}'
    H = all_y.shape[-1]
    per = []
    for h in range(H):
        m = _compute_metrics(all_y[:, :, h], all_pred[:, :, h])
        m['h'] = h + 1
        per.append(m)
    return per


@torch.no_grad()
def evaluate_baseline(model: nn.Module, loader: DataLoader,
                      adj, device: str = 'cpu',
                      per_horizon: bool = False) -> dict:
    """
    adj=None → LSTM/GRU (그래프 없음).

    [멀티스텝]
      모델 출력이 (B, N, H) 든 (B, N) 든 그대로 모아 _compute_metrics 로 채점.
      per_horizon=True 이고 출력이 (B,N,H) 이면, 반환 dict 에
      'per_horizon' 키로 horizon별 지표 리스트를 추가한다.
      (학습 중 val 평가는 per_horizon=False 로 가볍게, 최종 test 만 True 권장)
    """
    model.eval()
    ay, ap = [], []
    for x_r, _, _, _, y in loader:
        pred = model(x_r.to(device), adj) if adj is not None else model(x_r.to(device))
        ay.append(y.numpy())
        ap.append(pred.cpu().numpy())
    all_y    = np.concatenate(ay)
    all_pred = np.concatenate(ap)

    m = _compute_metrics(all_y, all_pred)
    if per_horizon and all_y.ndim == 3:
        m['per_horizon'] = _compute_metrics_per_horizon(all_y, all_pred)
    return m


def print_bl_metrics(m: dict, name: str = 'Baseline'):
    """기존 print_metrics() 와 동일한 포맷."""
    print(f'\n── {name} 평가 결과 ────────────────────────────────────')
    print(f'  [학습 진단]')
    print(f'  Zero rate (실제) : {m["zero_rate_true"]:.4f}')
    print(f'  Zero rate (예측) : {m["zero_rate_pred"]:.4f}')
    print(f'  KL Divergence    : {m["kl_div"]:.4f}')
    print(f'  [회귀 지표]')
    print(f'  MAE              : {m["mae"]:.4f}')
    print(f'  RMSE             : {m["rmse"]:.4f}')
    print(f'  MAE@positive     : {m["mae_pos"]:.4f}')
    print(f'  RMSE@positive    : {m["rmse_pos"]:.4f}')
    print(f'  [분류 지표]')
    print(f'  Recall@positive  : {m["recall_pos"]:.4f}')
    print(f'  Precision@pos    : {m["precision_pos"]:.4f}')
    print(f'  F1@positive      : {m["f1_pos"]:.4f}')
    print(f'  [랭킹 지표]')
    print(f'  HR@20            : {m["hr_at_20"]:.4f}')


def results_table(all_results: dict, sort_by: str = 'HR@20',
                  ascending: bool = False) -> pd.DataFrame:
    rows = []
    for name, m in all_results.items():
        rows.append({
            'Model':      name,
            'MAE':        round(m['mae'],                         4),
            'RMSE':       round(m['rmse'],                        4),
            'MAE@pos':    round(m.get('mae_pos',  float('nan')),  4),
            'RMSE@pos':   round(m.get('rmse_pos', float('nan')),  4),
            'Recall@pos': round(m['recall_pos'],                  4),
            'Prec@pos':   round(m['precision_pos'],               4),
            'F1@pos':     round(m['f1_pos'],                      4),
            'HR@20':      round(m['hr_at_20'],                    4),
            'ZR_pred':    round(m.get('zero_rate_pred', float('nan')), 4),
        })
    df = (pd.DataFrame(rows).set_index('Model')
          .sort_values(sort_by, ascending=ascending))
    print('\n' + '='*90)
    print('📊  Baseline Comparison — Test Set')
    print('='*90)
    print(df.to_string())
    print('='*90)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  BL-0C  [멀티스텝] horizon별 결과 표 / degradation curve
# ══════════════════════════════════════════════════════════════════════════════

def horizon_table(all_results: dict, metric: str = 'mae') -> pd.DataFrame:
    """
    [멀티스텝] 모델 × horizon 표를 만든다 (행=모델, 열=h1..hH, 값=지정 지표).
    per_horizon 을 저장한 모델만 포함된다.
    metric: 'mae','rmse','mae_pos','f1_pos','hr_at_20' 등 _compute_metrics 키.
    """
    rows = {}
    for name, m in all_results.items():
        ph = m.get('per_horizon')
        if not ph:
            continue
        rows[name] = {f'h{d["h"]}': round(d.get(metric, float("nan")), 4) for d in ph}
    if not rows:
        print('⚠️ per_horizon 결과가 있는 모델이 없습니다.')
        return pd.DataFrame()
    df = pd.DataFrame(rows).T
    df = df[[f'h{i}' for i in range(1, H_HORIZON + 1) if f'h{i}' in df.columns]]
    print('\n' + '='*90)
    print(f'📈  Horizon별 {metric.upper()} — 낮을수록 좋음(오차) / 높을수록 좋음(f1·hr)')
    print('='*90)
    print(df.to_string())
    print('='*90)
    return df


def plot_degradation(all_results: dict, metric: str = 'mae', ax=None):
    """
    [멀티스텝] horizon(x) vs 지표(y) 곡선 — 모델별로 겹쳐 그린다.
    리뷰어 W2 대응 핵심 그림: '예측 지평이 늘수록 정확도가 어떻게 변하는가'.
    matplotlib 필요.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for name, m in all_results.items():
        ph = m.get('per_horizon')
        if not ph:
            continue
        hs = [d['h'] for d in ph]
        ys = [d.get(metric, float('nan')) for d in ph]
        ax.plot(hs, ys, marker='o', markersize=3, label=name)
    ax.set_xlabel('Forecast horizon (hours ahead)')
    ax.set_ylabel(metric.upper())
    ax.set_title(f'{metric.upper()} vs. forecast horizon')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return ax


def report_wide(all_results: dict,
                horizons=(1, 3, 6),
                model_order=None) -> 'pd.DataFrame':
    """
    [멀티스텝] 논문용 wide 표.
      행   = 모델
      열   = (1hour / 3hours / 6hours) × (MAE, RMSE, Non-zero MAE, F1, HR@20%)
    per_horizon 이 저장된 모델만 포함된다.

    사용:
      results = load_bl_results()
      # MCAST 결과도 같은 형식으로 넣어둘 것:
      #   per_h, overall = evaluate_multistep(model, test_loader, SC, KC, AP, AA, AV)
      #   results['MCAST-ZINB'] = {**overall, 'per_horizon': per_h}
      df = report_wide(results, horizons=(1, 3, 6),
                       model_order=['HA','LSTM','GRU','STGCN','DCRNN',
                                    'GraphWaveNet','ASTGCN','MCAST-NB','MCAST-ZINB'])
      print(df.to_string())
      save_report_wide(df, 'horizon_report')   # csv/tex/xlsx 로 저장
    """
    # (표시이름, per_horizon 딕셔너리 키)
    metric_map = [
        ('MAE',          'mae'),
        ('RMSE',         'rmse'),
        ('Non-zero MAE', 'mae_pos'),
        ('F1',           'f1_pos'),
        ('HR@20%',       'hr_at_20'),
    ]
    hname = lambda h: f'{h}hour' if h == 1 else f'{h}hours'

    cols = pd.MultiIndex.from_tuples(
        [(hname(h), lab) for h in horizons for lab, _ in metric_map])

    names = model_order if model_order is not None else list(all_results.keys())
    data = {}
    for name in names:
        m = all_results.get(name)
        if not m:
            continue
        ph = m.get('per_horizon')
        if not ph:
            print(f'⚠️ {name}: per_horizon 없음 → 표에서 제외 '
                  f'(멀티스텝 재평가 필요)')
            continue
        by_h = {d['h']: d for d in ph}          # h번호로 찾기 (리스트 인덱스 아님)
        vals = []
        for h in horizons:
            d = by_h.get(h, {})
            if not d:
                print(f'⚠️ {name}: h={h} 결과 없음 (H_HORIZON < {h}?)')
            for _, key in metric_map:
                vals.append(round(d.get(key, float('nan')), 4))
        data[name] = vals

    if not data:
        print('⚠️ 표에 넣을 모델이 없습니다. per_horizon 저장 여부를 확인하세요.')
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data, orient='index', columns=cols)
    df.index.name = 'Model'

    # 경고: 서로 다른 horizon 인데 값이 완전히 동일하면 멀티스텝이 아닐 수 있음
    if len(horizons) >= 2:
        first, last = hname(horizons[0]), hname(horizons[-1])
        if first in df.columns.get_level_values(0) and last in df.columns.get_level_values(0):
            same = (df[first].round(4).values == df[last].round(4).values).all()
            if same:
                print(f'⚠️ 주의: {first} 와 {last} 지표가 완전히 동일합니다. '
                      f'모델이 모든 horizon에 같은 값을 내고 있을 수 있어요 '
                      f'(출력층/reshape 또는 표 생성 로직 점검 권장).')
    return df


def save_report_wide(df: 'pd.DataFrame', stem: str = 'horizon_report'):
    """report_wide 결과를 csv / tex / xlsx 로 저장."""
    if df is None or df.empty:
        print('⚠️ 빈 표 — 저장 생략')
        return
    df.to_csv(f'{stem}.csv')
    try:
        df.to_latex(f'{stem}.tex', escape=False, na_rep='-', multicolumn=True,
                    multicolumn_format='c')
    except Exception as e:
        print(f'(LaTeX 저장 건너뜀: {e})')
    try:
        df.to_excel(f'{stem}.xlsx')
    except Exception as e:
        print(f'(xlsx 저장 건너뜀 — openpyxl 필요: {e})')
    print(f'💾 저장: {stem}.csv / {stem}.tex / {stem}.xlsx')


# ══════════════════════════════════════════════════════════════════════════════
#  BL-0D  [복잡도] 파라미터 수 + 추론 시간(latency) 측정  — 리뷰어 cost 대응
# ══════════════════════════════════════════════════════════════════════════════

def count_params(model) -> int:
    """학습 가능한 파라미터 총 개수."""
    if not hasattr(model, 'parameters'):
        return 0                                   # HistoricalAverage 등 비신경망
    return int(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def measure_latency(forward_fn, n_warmup: int = 3, n_repeat: int = 20,
                    device: str = 'cpu') -> dict:
    """
    추론 시간 측정 (배치 1회 forward 기준).
      forward_fn: 인자 없이 호출하면 1회 forward 하는 함수 (lambda 로 감싸 전달)
      n_warmup:   측정 제외할 워밍업 실행 (첫 실행은 캐시/컴파일로 느림)
      n_repeat:   평균을 낼 반복 횟수
    GPU 는 비동기 실행이므로 synchronize 로 정확히 계측한다.
    반환: {'latency_ms_mean', 'latency_ms_std', 'n_repeat'}
    """
    import time
    is_cuda = ('cuda' in str(device))

    for _ in range(n_warmup):                       # 워밍업
        forward_fn()
    if is_cuda:
        torch.cuda.synchronize()

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        forward_fn()
        if is_cuda:
            torch.cuda.synchronize()               # GPU 완료까지 대기 후 계측
        times.append((time.perf_counter() - t0) * 1000.0)   # ms

    arr = np.array(times)
    return {'latency_ms_mean': float(arr.mean()),
            'latency_ms_std':  float(arr.std()),
            'n_repeat':        n_repeat}


def measure_cost(model, forward_fn, device: str = 'cpu',
                 n_warmup: int = 3, n_repeat: int = 20) -> dict:
    """
    한 모델의 비용 지표: 파라미터 수 + 추론 시간.
      forward_fn: 인자 없이 호출 시 model 이 배치 1개를 forward 하도록 감싼 함수.
                  모델마다 forward 시그니처가 달라 호출부에서 lambda 로 맞춰 전달한다.
    """
    if hasattr(model, 'eval'):
        model.eval()
    n_param = count_params(model)
    lat     = measure_latency(forward_fn, n_warmup, n_repeat, device)
    return {'n_params': n_param, **lat}


def cost_table(cost_results: dict, model_order=None) -> 'pd.DataFrame':
    """
    모델별 비용 표.  cost_results: {모델이름: measure_cost 반환 dict}.
    열: Params, Params(M), Latency(ms), Latency std(ms).
    성능 표(report_wide) 옆에 나란히 싣기 좋은 형태.
    """
    names = model_order if model_order is not None else list(cost_results.keys())
    rows = []
    for name in names:
        c = cost_results.get(name)
        if not c:
            continue
        rows.append({
            'Model':          name,
            'Params':         c.get('n_params', 0),
            'Params(M)':      round(c.get('n_params', 0) / 1e6, 3),
            'Latency(ms)':    round(c.get('latency_ms_mean', float('nan')), 3),
            'Latency±(ms)':   round(c.get('latency_ms_std',  float('nan')), 3),
        })
    if not rows:
        print('⚠️ 비용 표에 넣을 모델이 없습니다.')
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index('Model')
    print('\n' + '='*70)
    print('🧮  Model Complexity — Params & Inference Latency')
    print('='*70)
    print(df.to_string())
    print('='*70)
    return df


def build_forward_fn(model, sample_batch, adj=None, device: str = 'cpu',
                     mcast_args: dict = None):
    """
    measure_cost 에 넘길 forward_fn 을 만들어 준다 (호출부 편의용 헬퍼).

    두 가지 케이스:
    ── baseline (LSTM/GRU/STGCN/DCRNN/GWN/ASTGCN) ──
        sample_batch = (x_r, ...) 튜플 중 x_r 만 사용.
        adj=None → 그래프 없는 모델(LSTM/GRU), adj 전달 → 그래프 모델.
        build_forward_fn(model, batch, adj=AP, device=device)

    ── MCAST (forward 시그니처가 다름) ──
        mcast_args 에 나머지 인자를 넘긴다:
        build_forward_fn(model, batch, device=device, mcast_args=dict(
            static_context=SC, keyword_context=KC,
            adj_phys=AP, adj_adapt=AA, adj_visual=AV))
        → model(x_r, x_d, x_w, temporal, SC, KC, AP, AA, AV) 형태로 호출.
    """
    x_r, x_d, x_w, temporal, _ = sample_batch
    x_r = x_r.to(device)

    if mcast_args is not None:
        x_d = x_d.to(device); x_w = x_w.to(device); temporal = temporal.to(device)
        def fn():
            model(x_r, x_d, x_w, temporal,
                  mcast_args['static_context'], mcast_args['keyword_context'],
                  mcast_args['adj_phys'], mcast_args['adj_adapt'], mcast_args['adj_visual'])
        return fn

    if adj is not None:
        def fn():
            model(x_r, adj)
        return fn
    def fn():
        model(x_r)
    return fn


# ══════════════════════════════════════════════════════════════════════════════
#  BL-0B  체크포인트 & 학습 유틸
# ══════════════════════════════════════════════════════════════════════════════

def save_bl_results(results: dict, path: str = BL_RESULTS_PATH):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'💾 결과 저장: {path} | 완료 모델: {list(results.keys())}')


def load_bl_results(path: str = BL_RESULTS_PATH) -> dict:
    if Path(path).exists():
        with open(path, encoding='utf-8') as f:
            res = json.load(f)
        print(f'📂 결과 로드: {path} | 완료 모델: {list(res.keys())}')
        return res
    return {}


def print_progress(results_path: str = BL_RESULTS_PATH, ckpt_dir: str = None):
    """완료 / 체크포인트 저장 / 미시작 현황."""
    if ckpt_dir is None:
        ckpt_dir = BL_CKPT_DIR
    results = {}
    if Path(results_path).exists():
        with open(results_path, encoding='utf-8') as f:
            results = json.load(f)
    print('\n📋 Baseline 실험 현황')
    print('─' * 60)
    for name in _MODEL_ORDER:
        if name in results:
            m = results[name]
            print(f'  ✅ {name:<16} MAE={m["mae"]:.4f}  HR@20={m["hr_at_20"]:.4f}')
        else:
            ckpt = Path(ckpt_dir) / f'ckpt_{name}.pt'
            if ckpt.exists():
                try:
                    ck = torch.load(str(ckpt), map_location='cpu')
                    ep = ck.get('epoch', '?')
                    bm = ck.get('best_mae', float('nan'))
                    print(f'  ⏩ {name:<16} 체크포인트 {ep}epoch | best_mae={bm:.4f}')
                except Exception:
                    print(f'  ⏩ {name:<16} 체크포인트 존재 (손상 가능성)')
            else:
                print(f'  ⬜ {name:<16} 미시작')
    done      = [n for n in _MODEL_ORDER if n in results]
    remaining = [n for n in _MODEL_ORDER if n not in results]
    print('─' * 60)
    print(f'  완료: {len(done)}/7  |  남음: {remaining}')


def train_bl_with_checkpoint(
    model:           nn.Module,
    loader_tr:       DataLoader,
    loader_va:       DataLoader,
    adj,
    model_name:      str,
    device:          str,
    n_epochs:        int,
    lr:              float,
    checkpoint_path: str,
    save_every:      int = BL_SAVE_EVERY,
    pos_weight:      float = BL_POS_WEIGHT,
) -> nn.Module:
    """
    체크포인트 재개 지원 학습 루프.

    손실함수: Weighted MSE
      w = pos_weight (target>0) else 1
      loss = mean(w * (pred - target)²)
      → zero_rate=99.56% 환경에서 zero collapse 방지

    체크포인트: save_every epoch마다 + 마지막 epoch (Drive 저장 시 런타임 재시작 후 재개)
    """
    ckpt_path = Path(checkpoint_path)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5)

    start_epoch = 0
    best_mae    = float('inf')
    best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── 체크포인트 로드 (재개) ──────────────────────────────────────────────
    if ckpt_path.exists():
        ck = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict({k: v.to(device) for k, v in ck['model_state'].items()})
        optimizer.load_state_dict(ck['optimizer_state'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        scheduler.load_state_dict(ck['scheduler_state'])
        start_epoch = ck['epoch']
        best_mae    = ck['best_mae']
        best_state  = ck['best_model_state']
        print(f'⏩ [{model_name}] 재개: epoch {start_epoch+1}/{n_epochs} | '
              f'best_val_mae={best_mae:.4f}')

    if start_epoch >= n_epochs:
        print(f'✅ [{model_name}] 이미 완료 (epoch {n_epochs}/{n_epochs})')
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        return model

    n_left = n_epochs - start_epoch
    print(f'\n🚀 [{model_name}] 학습 | epoch {start_epoch+1}~{n_epochs} ({n_left}회 남음) '
          f'| lr={lr} | pos_weight={pos_weight} '
          f'| params={sum(p.numel() for p in model.parameters()):,}')

    for epoch in range(start_epoch + 1, n_epochs + 1):
        model.train()
        total_loss = 0.0

        for x_r, _, _, _, target in loader_tr:
            x_r    = x_r.to(device)
            target = target.to(device)
            pred   = model(x_r, adj) if adj is not None else model(x_r)

            # ── Weighted MSE ───────────────────────────────────────────────
            w    = torch.where(target > 0,
                               target.new_full(target.shape, pos_weight),
                               target.new_ones(target.shape))
            loss = (F.mse_loss(pred, target, reduction='none') * w).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        val_m = evaluate_baseline(model, loader_va, adj, device)
        scheduler.step(val_m['mae'])

        if val_m['mae'] < best_mae:
            best_mae   = val_m['mae']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == start_epoch + 1 or epoch == n_epochs:
            print(f'  Epoch {epoch:>3}/{n_epochs} | '
                  f'wMSE={total_loss/len(loader_tr):.4f} | '
                  f'Val MAE={val_m["mae"]:.4f} | '
                  f'ZR_pred={val_m["zero_rate_pred"]:.4f} | '
                  f'HR@20={val_m["hr_at_20"]:.4f}')

        if epoch % save_every == 0 or epoch == n_epochs:
            torch.save({
                'epoch':            epoch,
                'model_state':      {k: v.cpu() for k, v in model.state_dict().items()},
                'optimizer_state':  optimizer.state_dict(),
                'scheduler_state':  scheduler.state_dict(),
                'best_mae':         best_mae,
                'best_model_state': best_state,
            }, str(ckpt_path))

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    print(f'✅ [{model_name}] 완료 | Best Val MAE = {best_mae:.4f}')
    return model


def _gnn_loader(base: DataLoader) -> DataLoader:
    return DataLoader(base.dataset, batch_size=BL_BATCH_SIZE_GNN,
                      shuffle=False, num_workers=0, pin_memory=False)


def _header(title: str):
    print('\n' + '='*62 + '\n' + title + '\n' + '='*62)


# ══════════════════════════════════════════════════════════════════════════════
#  BL-1 : Historical Average
# ══════════════════════════════════════════════════════════════════════════════

class HistoricalAverage:
    """요일 × 시간대 평균 예측 (7×24=168 패턴). 학습/역전파 없음."""

    def __init__(self):
        self.ha_table: dict = {}
        self.N: int = 0

    def fit(self, pivot_df: pd.DataFrame,
            train_start: str, train_end: str) -> 'HistoricalAverage':
        cols  = pivot_df.columns
        mask  = (cols >= pd.to_datetime(train_start)) & (cols <= pd.to_datetime(train_end))
        train = pivot_df.loc[:, mask]
        self.N = pivot_df.shape[0]
        accum: dict = {}
        for ts in train.columns:
            key = (ts.weekday(), ts.hour)
            accum.setdefault(key, []).append(train[ts].values.astype(np.float32))
        for key, vlist in accum.items():
            self.ha_table[key] = np.mean(np.stack(vlist, axis=1), axis=1)
        print(f'✅ HA fit | 패턴 {len(self.ha_table)}/168 | N={self.N:,}')
        return self

    def predict_at(self, t: pd.Timestamp) -> np.ndarray:
        return self.ha_table.get((t.weekday(), t.hour),
                                 np.zeros(self.N, dtype=np.float32))

    def evaluate(self, pivot_df: pd.DataFrame,
                 test_start: str, test_end: str) -> dict:
        """[단일스텝 원본] 유지 — 하위 호환용."""
        cols    = pivot_df.columns
        mask    = (cols >= pd.to_datetime(test_start)) & (cols <= pd.to_datetime(test_end))
        test_ts = cols[mask]
        if len(test_ts) == 0:
            raise ValueError(f'테스트 기간에 타임스텝 없음: {test_start}~{test_end}')
        all_y    = pivot_df.loc[:, mask].values.T.astype(np.float32)
        all_pred = np.stack([self.predict_at(t) for t in test_ts])
        return _compute_metrics(all_y, all_pred)

    def evaluate_multistep(self, pivot_df: pd.DataFrame,
                           test_start: str, test_end: str,
                           h_recent: int, h_weekly: int,
                           per_horizon: bool = True) -> dict:
        """
        [멀티스텝] 각 기준시점 t 에서 미래 H_HORIZON 시간을 예측.
          예측: t+h 시각의 (요일,시각) 과거평균  (h=0..H-1)
          정답: pivot_df 의 t..t+H-1 실제값
        신경망 baseline(CSTDataset) 과 '동일한 시점 집합' 을 쓰도록
        앞(min_offset)·뒤(미래 H칸이 구간 안) 경계를 똑같이 적용한다.
        → 모든 모델이 같은 (T, N, H) 로 채점되어 비교가 공정해진다.
        """
        cols       = pivot_df.columns
        n_times    = len(cols)
        start, end = pd.to_datetime(test_start), pd.to_datetime(test_end)
        min_offset = 168 + h_weekly                          # CSTDataset 과 동일

        base_idx = [
            i for i, t in enumerate(cols)
            if start <= t <= end
            and i >= min_offset
            and i + H_HORIZON <= n_times
            and cols[i + H_HORIZON - 1] <= end
        ]
        if len(base_idx) == 0:
            raise ValueError(f'유효 시점 없음: {test_start}~{test_end}')

        data = pivot_df.values.astype(np.float32)            # (N, 전체시간)
        y_list, p_list = [], []
        for i in base_idx:
            # 정답: (N, H)
            y_list.append(data[:, i:i + H_HORIZON])
            # 예측: 각 미래 시각의 요일×시각 평균 → (N, H)
            preds_h = [self.predict_at(cols[i + h]) for h in range(H_HORIZON)]
            p_list.append(np.stack(preds_h, axis=1))         # (N, H)

        all_y    = np.stack(y_list, axis=0)                  # (T, N, H)
        all_pred = np.stack(p_list, axis=0)                  # (T, N, H)

        m = _compute_metrics(all_y, all_pred)
        if per_horizon:
            m['per_horizon'] = _compute_metrics_per_horizon(all_y, all_pred)
        return m


# ══════════════════════════════════════════════════════════════════════════════
#  BL-2 : LSTM Baseline
# ══════════════════════════════════════════════════════════════════════════════

class LSTMBaseline(nn.Module):
    """
    노드 독립 LSTM (weight sharing across nodes).
    x_r: (B,1,N,T) → (B*N,T,1) → LSTM → FC → softplus → (B,N)
    출력: F.softplus — 항상 양수, gradient 항상 존재 (dead neuron 없음)
    """
    def __init__(self, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_dim, n_layers, batch_first=True,
                            dropout=0.1 if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, H_HORIZON)   # [멀티스텝] 1 → H_HORIZON

    def forward(self, x_r: torch.Tensor, adj=None) -> torch.Tensor:
        B, _, N, T = x_r.shape
        x      = x_r.squeeze(1).permute(0, 2, 1).reshape(B * N, T, 1)
        out, _ = self.lstm(x)
        # (B*N, H) → (B, N, H)   [멀티스텝] reshape 마지막 축을 H_HORIZON 으로
        return F.softplus(self.fc(out[:, -1])).reshape(B, N, H_HORIZON)


# ══════════════════════════════════════════════════════════════════════════════
#  BL-3 : GRU Baseline
# ══════════════════════════════════════════════════════════════════════════════

class GRUBaseline(nn.Module):
    """
    노드 독립 GRU (weight sharing across nodes).
    출력: F.softplus
    """
    def __init__(self, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(1, hidden_dim, n_layers, batch_first=True,
                          dropout=0.1 if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim, H_HORIZON)   # [멀티스텝] 1 → H_HORIZON

    def forward(self, x_r: torch.Tensor, adj=None) -> torch.Tensor:
        B, _, N, T = x_r.shape
        x      = x_r.squeeze(1).permute(0, 2, 1).reshape(B * N, T, 1)
        out, _ = self.gru(x)
        # (B*N, H) → (B, N, H)   [멀티스텝]
        return F.softplus(self.fc(out[:, -1])).reshape(B, N, H_HORIZON)


# ══════════════════════════════════════════════════════════════════════════════
#  BL-4 : STGCN Baseline
# ══════════════════════════════════════════════════════════════════════════════

class STConvBlock(nn.Module):
    """
    STGCN-style ST Block.
    Gated-TCN → GCN(last timestep) → Residual + LayerNorm.
    GCN을 마지막 타임스텝에만 적용 후 broadcast add
    → N=5,661에서 전 타임스텝 GCN 대비 메모리 대폭 절감.
    """
    def __init__(self, in_c: int, out_c: int, k_t: int = 3):
        super().__init__()
        pad        = k_t // 2
        self.tcn_f = nn.Conv2d(in_c, out_c, (1, k_t), padding=(0, pad))
        self.tcn_g = nn.Conv2d(in_c, out_c, (1, k_t), padding=(0, pad))
        self.gcn   = nn.Linear(out_c, out_c, bias=False)
        self.res   = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.norm  = nn.LayerNorm(out_c)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h   = torch.tanh(self.tcn_f(x)) * torch.sigmoid(self.tcn_g(x))   # (B,out_c,N,T)
        hl  = h[..., -1].permute(0, 2, 1)                                  # (B,N,out_c)
        hg  = F.relu(torch.matmul(adj, self.gcn(hl)))                      # (B,N,out_c)
        h   = h + hg.permute(0, 2, 1).unsqueeze(-1)
        res = self.res(x)
        out_last = self.norm((h + res)[..., -1].permute(0, 2, 1))
        return torch.cat([(h + res)[..., :-1],
                          out_last.permute(0, 2, 1).unsqueeze(-1)], dim=-1)


class STGCNBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.block1 = STConvBlock(1, hidden_dim)
        self.block2 = STConvBlock(hidden_dim, hidden_dim)
        self.fc     = nn.Linear(hidden_dim, H_HORIZON)   # [멀티스텝] 1 → H_HORIZON

    def forward(self, x_r: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.block2(self.block1(x_r, adj), adj)
        # fc 출력이 (B, N, H) → squeeze(-1) 제거   [멀티스텝]
        return F.softplus(self.fc(h[..., -1].permute(0, 2, 1)))


# ══════════════════════════════════════════════════════════════════════════════
#  BL-5 : DCRNN Baseline
# ══════════════════════════════════════════════════════════════════════════════

class DiffConv(nn.Module):
    """K-hop Diffusion Conv: linear(concat[x, A@x, A²@x, …, A^K@x])."""
    def __init__(self, in_dim: int, out_dim: int, K: int = 2):
        super().__init__()
        self.K      = K
        self.linear = nn.Linear((K + 1) * in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        parts = [x]
        h = x
        for _ in range(self.K):
            h = torch.matmul(adj, h)
            parts.append(h)
        return self.linear(torch.cat(parts, dim=-1))


class DCGRUCell(nn.Module):
    """Diffusion Convolutional GRU Cell (Li et al., 2018)."""
    def __init__(self, in_dim: int, hidden_dim: int, K: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        xh = in_dim + hidden_dim
        self.diff_r = DiffConv(xh, hidden_dim, K)
        self.diff_u = DiffConv(xh, hidden_dim, K)
        self.diff_c = DiffConv(xh, hidden_dim, K)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        xh = torch.cat([x, h], dim=-1)
        r  = torch.sigmoid(self.diff_r(xh, adj))
        u  = torch.sigmoid(self.diff_u(xh, adj))
        c  = torch.tanh(self.diff_c(torch.cat([x, r * h], dim=-1), adj))
        return u * h + (1.0 - u) * c


class DCRNNBaseline(nn.Module):
    """
    DCRNN Baseline (Li et al., 2018): 1-layer DCGRU, K=2.
    K=2: 평균 차수 4.5 기준 2-hop 이웃 포착.
    """
    def __init__(self, hidden_dim: int = 64, K: int = 2, T_dcrnn: int = 12):
        super().__init__()
        self.T_dcrnn = T_dcrnn
        self.cell    = DCGRUCell(1, hidden_dim, K)
        self.fc      = nn.Linear(hidden_dim, H_HORIZON)   # [멀티스텝] 1 → H_HORIZON

    def forward(self, x_r: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, _, N, _ = x_r.shape
        x_seq = x_r.squeeze(1)[..., -self.T_dcrnn:]                    # (B,N,T)
        h = torch.zeros(B, N, self.cell.hidden_dim, device=x_r.device)
        for t in range(self.T_dcrnn):
            h = self.cell(x_seq[..., t].unsqueeze(-1), h, adj)
        # fc(h): (B, N, H) → squeeze(-1) 제거   [멀티스텝]
        return F.softplus(self.fc(h))


# ══════════════════════════════════════════════════════════════════════════════
#  BL-6 : Graph WaveNet Baseline
# ══════════════════════════════════════════════════════════════════════════════

class WaveNetLayer(nn.Module):
    """
    WaveNet-style dilated causal conv + 이중 GCN(last step) + skip.

    그래프 컨볼루션:
      gcn_phys:  adj_phys (물리적 인접, 고정) 적용
      gcn_adapt: adj_adapt (학습 가능한 적응형 인접) 적용
      출력 = relu(A_phys @ W_phys(h) + A_adapt @ W_adapt(h))
    → Wu et al. (2019) Graph WaveNet 원논문 구조 재현
    """
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv_f    = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation))
        self.conv_g    = nn.Conv2d(channels, channels, (1, 2), dilation=(1, dilation))
        self.gcn_phys  = nn.Linear(channels, channels, bias=False)
        self.gcn_adapt = nn.Linear(channels, channels, bias=False)
        self.res       = nn.Conv2d(channels, channels, 1)
        self.skip      = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor,
                adj_phys: torch.Tensor,
                adj_adapt: torch.Tensor):
        h     = torch.tanh(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))
        T_out = h.size(-1)
        hl    = h[..., -1].permute(0, 2, 1)                              # (B,N,C)
        hg    = F.relu(
            torch.matmul(adj_phys,  self.gcn_phys(hl)) +
            torch.matmul(adj_adapt, self.gcn_adapt(hl))
        )
        skip  = self.skip(hg.permute(0, 2, 1).unsqueeze(-1))
        return h + self.res(x[..., -T_out:]), skip


class GraphWaveNetBaseline(nn.Module):
    """
    Graph WaveNet (Wu et al., 2019) — 원논문 완전 재현.

    인접행렬:
      adj_phys  : 물리적 도로 인접 (고정, 노트북에서 전달)
      adj_adapt : 학습 가능한 적응형 인접
                  A_adapt = softmax(relu(E1 @ E2^T))
                  E1, E2 ∈ R^{N × emb_dim=10}
    레이어: 4-dilated (d=1,2,4,8), skip connection 합산
    T 변화: 24 → 23 → 21 → 17 → 9 (dilation으로 축소)
    """
    def __init__(self, N: int, hidden_dim: int = 64, emb_dim: int = 10):
        super().__init__()
        self.input_proj = nn.Conv2d(1, hidden_dim, 1)
        self.nodevec1   = nn.Parameter(torch.randn(N, emb_dim))
        self.nodevec2   = nn.Parameter(torch.randn(N, emb_dim))
        self.layers     = nn.ModuleList(
            [WaveNetLayer(hidden_dim, d) for d in [1, 2, 4, 8]])
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, H_HORIZON)   # [멀티스텝] 1 → H_HORIZON

    def forward(self, x_r: torch.Tensor, adj_phys: torch.Tensor) -> torch.Tensor:
        # 적응형 인접행렬: forward당 1회 계산, 4 레이어 공유
        A_adapt = F.softmax(
            F.relu(self.nodevec1 @ self.nodevec2.t()), dim=-1)         # (N,N)

        x = self.input_proj(x_r)
        skip_sum = None
        for layer in self.layers:
            x, skip  = layer(x, adj_phys, A_adapt)
            skip_sum = skip if skip_sum is None else skip_sum + skip

        out = F.relu(self.fc1(skip_sum[..., 0].permute(0, 2, 1)))
        # fc2 출력이 (B, N, H) → squeeze(-1) 제거   [멀티스텝]
        return F.softplus(self.fc2(out))


# ══════════════════════════════════════════════════════════════════════════════
#  BL-7 : ASTGCN Baseline
# ══════════════════════════════════════════════════════════════════════════════

class SpatialAttention(nn.Module):
    """
    Low-rank Spatial Attention (Guo et al., 2019 ASTGCN 적용).

    원논문: V ∈ R^{N×N} (N=5,661이면 32M 파라미터)
    본 구현: Q/K/V를 d_attn=32 차원으로 low-rank projection
      → 파라미터 대폭 절감, Flash Attention 활용 가능
      → 논문 기재 권장: "N=5,661 규모 적용을 위한 low-rank 근사 (d=32)"

    메모리:
      GPU (Flash Attention, PyTorch≥2.0): S=(B,N,N) 행렬 미생성 → 효율적
      CPU: S=(B,N,N) 생성 → B=1, N=5661이면 128MB — GPU 강력 권장
    """
    def __init__(self, in_c: int, T_in: int, d_attn: int = 32):
        super().__init__()
        feat_dim  = in_c * T_in
        self.W_q  = nn.Linear(feat_dim, d_attn, bias=False)
        self.W_k  = nn.Linear(feat_dim, d_attn, bias=False)
        self.W_v  = nn.Linear(feat_dim, d_attn, bias=False)
        self.proj = nn.Linear(d_attn, in_c,    bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N, T = x.shape
        h   = x.permute(0, 2, 1, 3).reshape(B, N, C * T)  # (B,N,C*T)
        Q   = self.W_q(h)                                    # (B,N,d_attn)
        K   = self.W_k(h)
        V   = self.W_v(h)
        # GPU PyTorch≥2.0: Flash Attention (N×N 행렬 미생성)
        # CPU: 표준 attention (N×N 행렬 생성)
        out = F.scaled_dot_product_attention(Q, K, V)        # (B,N,d_attn)
        out = self.proj(out)                                  # (B,N,in_c)
        return out.permute(0, 2, 1).unsqueeze(-1)             # (B,in_c,N,1) — T broadcast용


class ASTGCNBlock(nn.Module):
    """
    ASTGCN Block (Guo et al., 2019) — Spatial Attention 포함 완전 구현.

    처리 순서:
      1. Spatial Attention (low-rank, residual add)
      2. Temporal Attention (학습 가능한 T×T 가중치)
      3. TCN (Gated Conv)
      4. GCN (마지막 타임스텝)
      5. Residual + LayerNorm
    """
    def __init__(self, in_c: int, out_c: int, T_in: int = 24,
                 k_t: int = 3, d_attn: int = 32):
        super().__init__()
        self.spat_att = SpatialAttention(in_c, T_in, d_attn)
        self.W_t      = nn.Parameter(torch.randn(T_in, T_in) / T_in**0.5)
        pad           = k_t // 2
        self.tcn      = nn.Conv2d(in_c, out_c, (1, k_t), padding=(0, pad))
        self.gcn      = nn.Linear(out_c, out_c, bias=False)
        self.res      = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.norm     = nn.LayerNorm(out_c)

    def forward(self, x_in: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, C, N, T = x_in.shape

        # 1. Spatial Attention (잔차 연결)
        h = x_in + self.spat_att(x_in)                         # (B,C,N,T)

        # 2. Temporal Attention
        att = F.softmax(self.W_t[:T, :T], dim=-1)
        h   = torch.matmul(h, att)                              # (B,C,N,T)

        # 3. TCN
        h2 = F.gelu(self.tcn(h))                               # (B,out_c,N,T)

        # 4. GCN — 마지막 타임스텝에만 적용 후 broadcast
        hl  = h2[..., -1].permute(0, 2, 1)                     # (B,N,out_c)
        hg  = F.relu(torch.matmul(adj, self.gcn(hl)))          # (B,N,out_c)

        # 5. Residual + LayerNorm
        res_l = self.res(x_in)[..., -1].permute(0, 2, 1)       # (B,N,out_c)
        out_l = self.norm(h2[..., -1].permute(0, 2, 1) + hg + res_l)
        return torch.cat([h2[..., :-1],
                          out_l.permute(0, 2, 1).unsqueeze(-1)], dim=-1)


class ASTGCNBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 64, T_in: int = 24):
        super().__init__()
        self.block1 = ASTGCNBlock(1, hidden_dim, T_in)
        self.block2 = ASTGCNBlock(hidden_dim, hidden_dim, T_in)
        self.fc     = nn.Linear(hidden_dim, H_HORIZON)   # [멀티스텝] 1 → H_HORIZON

    def forward(self, x_r: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.block2(self.block1(x_r, adj), adj)
        # fc 출력이 (B, N, H) → squeeze(-1) 제거   [멀티스텝]
        return F.softplus(self.fc(h[..., -1].permute(0, 2, 1)))


# ══════════════════════════════════════════════════════════════════════════════
#  BL-1 ~ BL-7  개별 실행 함수 (각각 독립 Colab 셀에 대응)
# ══════════════════════════════════════════════════════════════════════════════

def run_bl_ha(pivot_df, train_start, train_end, test_start, test_end,
              h_recent: int = 24, h_weekly: int = 24,
              results_path: str = BL_RESULTS_PATH):
    """
    BL-1: Historical Average.
    [멀티스텝] h_recent/h_weekly 는 CSTDataset 과 동일한 값(본 노트북의
    H_RECENT/H_WEEKLY)을 넘겨야 신경망 baseline 과 시점 집합이 일치한다.
    """
    results = load_bl_results(results_path)
    if 'HA' in results:
        print('✅ HA 이미 완료 — 결과 재사용')
        print_bl_metrics(results['HA'], 'HA')
        return results
    _header('BL-1 : Historical Average')
    ha = HistoricalAverage().fit(pivot_df, train_start, train_end)
    m  = ha.evaluate_multistep(pivot_df, test_start, test_end,
                               h_recent=h_recent, h_weekly=h_weekly,
                               per_horizon=True)               # [멀티스텝]
    print_bl_metrics(m, 'HA')
    results['HA'] = m
    save_bl_results(results, results_path)
    return results


def run_bl_lstm(train_loader, val_loader, test_loader,
                device: str = 'cpu', hidden_dim: int = 64,
                epochs: int = 70, lr: float = 5e-4,
                results_path: str = BL_RESULTS_PATH,
                ckpt_dir: str = None):
    """BL-2: LSTM (softplus + weighted MSE)."""
    if ckpt_dir is None: ckpt_dir = BL_CKPT_DIR
    results = load_bl_results(results_path)
    if 'LSTM' in results:
        print('✅ LSTM 이미 완료 — 결과 재사용')
        print_bl_metrics(results['LSTM'], 'LSTM')
        return results
    _header('BL-2 : LSTM')
    ckpt  = str(Path(ckpt_dir) / 'ckpt_LSTM.pt')
    model = LSTMBaseline(hidden_dim, 2).to(device)
    model = train_bl_with_checkpoint(model, train_loader, val_loader, None,
                                     'LSTM', device, epochs, lr, ckpt)
    m = evaluate_baseline(model, test_loader, None, device, per_horizon=True)
    print_bl_metrics(m, 'LSTM')
    results['LSTM'] = m
    save_bl_results(results, results_path)
    Path(ckpt).unlink(missing_ok=True)
    del model
    return results


def run_bl_gru(train_loader, val_loader, test_loader,
               device: str = 'cpu', hidden_dim: int = 64,
               epochs: int = 70, lr: float = 5e-4,
               results_path: str = BL_RESULTS_PATH,
               ckpt_dir: str = None):
    """BL-3: GRU (softplus + weighted MSE)."""
    if ckpt_dir is None: ckpt_dir = BL_CKPT_DIR
    results = load_bl_results(results_path)
    if 'GRU' in results:
        print('✅ GRU 이미 완료 — 결과 재사용')
        print_bl_metrics(results['GRU'], 'GRU')
        return results
    _header('BL-3 : GRU')
    ckpt  = str(Path(ckpt_dir) / 'ckpt_GRU.pt')
    model = GRUBaseline(hidden_dim, 2).to(device)
    model = train_bl_with_checkpoint(model, train_loader, val_loader, None,
                                     'GRU', device, epochs, lr, ckpt)
    m = evaluate_baseline(model, test_loader, None, device, per_horizon=True)
    print_bl_metrics(m, 'GRU')
    results['GRU'] = m
    save_bl_results(results, results_path)
    Path(ckpt).unlink(missing_ok=True)
    del model
    return results


def run_bl_stgcn(train_loader, val_loader, test_loader, adj,
                 device: str = 'cpu', hidden_dim: int = 64,
                 epochs: int = 70, lr: float = 5e-4,
                 results_path: str = BL_RESULTS_PATH,
                 ckpt_dir: str = None):
    """BL-4: STGCN (softplus + weighted MSE)."""
    if ckpt_dir is None: ckpt_dir = BL_CKPT_DIR
    results = load_bl_results(results_path)
    if 'STGCN' in results:
        print('✅ STGCN 이미 완료 — 결과 재사용')
        print_bl_metrics(results['STGCN'], 'STGCN')
        return results
    _header('BL-4 : STGCN')
    ckpt = str(Path(ckpt_dir) / 'ckpt_STGCN.pt')
    tr_g, va_g, te_g = _gnn_loader(train_loader), _gnn_loader(val_loader), _gnn_loader(test_loader)
    model = STGCNBaseline(hidden_dim).to(device)
    model = train_bl_with_checkpoint(model, tr_g, va_g, adj,
                                     'STGCN', device, epochs, lr, ckpt)
    m = evaluate_baseline(model, te_g, adj, device, per_horizon=True)
    print_bl_metrics(m, 'STGCN')
    results['STGCN'] = m
    save_bl_results(results, results_path)
    Path(ckpt).unlink(missing_ok=True)
    del model
    return results


def run_bl_dcrnn(train_loader, val_loader, test_loader, adj,
                 device: str = 'cpu', hidden_dim: int = 64,
                 epochs: int = 70, lr: float = 5e-4,
                 K: int = 2, T_dcrnn: int = 12,
                 results_path: str = BL_RESULTS_PATH,
                 ckpt_dir: str = None):
    """BL-5: DCRNN (softplus + weighted MSE)."""
    if ckpt_dir is None: ckpt_dir = BL_CKPT_DIR
    results = load_bl_results(results_path)
    if 'DCRNN' in results:
        print('✅ DCRNN 이미 완료 — 결과 재사용')
        print_bl_metrics(results['DCRNN'], 'DCRNN')
        return results
    _header('BL-5 : DCRNN')
    ckpt = str(Path(ckpt_dir) / 'ckpt_DCRNN.pt')
    tr_g, va_g, te_g = _gnn_loader(train_loader), _gnn_loader(val_loader), _gnn_loader(test_loader)
    model = DCRNNBaseline(hidden_dim, K=K, T_dcrnn=T_dcrnn).to(device)
    model = train_bl_with_checkpoint(model, tr_g, va_g, adj,
                                     'DCRNN', device, epochs, lr, ckpt)
    m = evaluate_baseline(model, te_g, adj, device, per_horizon=True)
    print_bl_metrics(m, 'DCRNN')
    results['DCRNN'] = m
    save_bl_results(results, results_path)
    Path(ckpt).unlink(missing_ok=True)
    del model
    return results


def run_bl_gwn(train_loader, val_loader, test_loader, adj,
               device: str = 'cpu', hidden_dim: int = 64,
               epochs: int = 70, lr: float = 5e-4,
               results_path: str = BL_RESULTS_PATH,
               ckpt_dir: str = None):
    """
    BL-6: Graph WaveNet (Wu et al., 2019) — 원논문 완전 재현.
    adj_phys + adaptive adj (nodevec1/2 학습) 사용.
    """
    if ckpt_dir is None: ckpt_dir = BL_CKPT_DIR
    results = load_bl_results(results_path)
    if 'GraphWaveNet' in results:
        print('✅ GraphWaveNet 이미 완료 — 결과 재사용')
        print_bl_metrics(results['GraphWaveNet'], 'GraphWaveNet')
        return results
    _header('BL-6 : Graph WaveNet')
    ckpt = str(Path(ckpt_dir) / 'ckpt_GraphWaveNet.pt')
    tr_g, va_g, te_g = _gnn_loader(train_loader), _gnn_loader(val_loader), _gnn_loader(test_loader)
    N     = adj.shape[0]
    model = GraphWaveNetBaseline(N, hidden_dim).to(device)
    model = train_bl_with_checkpoint(model, tr_g, va_g, adj,
                                     'GraphWaveNet', device, epochs, lr, ckpt)
    m = evaluate_baseline(model, te_g, adj, device, per_horizon=True)
    print_bl_metrics(m, 'GraphWaveNet')
    results['GraphWaveNet'] = m
    save_bl_results(results, results_path)
    Path(ckpt).unlink(missing_ok=True)
    del model
    return results


def run_bl_astgcn(train_loader, val_loader, test_loader, adj,
                  device: str = 'cpu', hidden_dim: int = 64,
                  epochs: int = 70, lr: float = 5e-4, T_in: int = 24,
                  results_path: str = BL_RESULTS_PATH,
                  ckpt_dir: str = None):
    """
    BL-7: ASTGCN (Guo et al., 2019) — Spatial Attention 포함 완전 구현.

    [GPU 강력 권장]
    Spatial Attention: N=5,661 노드 간 attention
      GPU (Flash Attention): S=(B,N,N) 미생성 → 효율적
      CPU: S=(B,N,N) = batch당 128MB → 학습 수 시간 소요 예상
           → Colab: 런타임 유형 변경 > GPU 선택 후 재실행

    batch_size: GPU=BL_BATCH_SIZE_GNN, CPU=1 (자동 감지)
    """
    if ckpt_dir is None: ckpt_dir = BL_CKPT_DIR
    results = load_bl_results(results_path)
    if 'ASTGCN' in results:
        print('✅ ASTGCN 이미 완료 — 결과 재사용')
        print_bl_metrics(results['ASTGCN'], 'ASTGCN')
        return results
    _header('BL-7 : ASTGCN')

    is_cpu = (str(device) == 'cpu')
    if is_cpu:
        print('⚠️  WARNING: ASTGCN Spatial Attention on CPU (N=5,661)')
        print('   GPU 권장 — Colab: 런타임 > 런타임 유형 변경 > GPU 선택')
        print('   CPU 강행: batch_size=1 자동 적용, epoch당 수 시간 소요 예상')
        astgcn_bs = 1
    else:
        astgcn_bs = BL_BATCH_SIZE_GNN

    ckpt = str(Path(ckpt_dir) / 'ckpt_ASTGCN.pt')
    tr_g = DataLoader(train_loader.dataset, batch_size=astgcn_bs, shuffle=False, num_workers=0)
    va_g = DataLoader(val_loader.dataset,   batch_size=astgcn_bs, shuffle=False, num_workers=0)
    te_g = DataLoader(test_loader.dataset,  batch_size=astgcn_bs, shuffle=False, num_workers=0)
    model = ASTGCNBaseline(hidden_dim, T_in).to(device)
    model = train_bl_with_checkpoint(model, tr_g, va_g, adj,
                                     'ASTGCN', device, epochs, lr, ckpt)
    m = evaluate_baseline(model, te_g, adj, device, per_horizon=True)
    print_bl_metrics(m, 'ASTGCN')
    results['ASTGCN'] = m
    save_bl_results(results, results_path)
    Path(ckpt).unlink(missing_ok=True)
    del model
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Colab 셀 템플릿
# ══════════════════════════════════════════════════════════════════════════════
#
# ┌─ [공통 설정 셀] — 런타임 시작/재시작마다 실행 ─────────────────────────────
# │  from google.colab import drive
# │  drive.mount('/content/drive')
# │  exec(open('/content/drive/MyDrive/mcast_data/baseline_models.py').read())
# │
# │  BL_CKPT_DIR     = '/content/drive/MyDrive/mcast_data'   # ← Drive 경로 필수
# │  BL_RESULTS_PATH = '/content/drive/MyDrive/mcast_data/bl_results.json'
# │
# │  # [재실행 시] LSTM/GRU/STGCN zero collapse 결과 제거 (HA만 유지)
# │  # import json
# │  # with open(BL_RESULTS_PATH) as f: r = json.load(f)
# │  # r = {k: v for k, v in r.items() if k == 'HA'}
# │  # with open(BL_RESULTS_PATH, 'w') as f: json.dump(r, f)
# │
# │  print_progress()
# └────────────────────────────────────────────────────────────────────────────
#
# ┌─ [BL-1] ─── run_bl_ha(pivot_df, TRAIN_START, TRAIN_END, TEST_START, TEST_END, h_recent=H_RECENT, h_weekly=H_WEEKLY)
# ┌─ [BL-2] ─── run_bl_lstm(train_loader, val_loader, test_loader, device=device, hidden_dim=HIDDEN_DIM, epochs=EPOCHS, lr=LR)
# ┌─ [BL-3] ─── run_bl_gru(train_loader, val_loader, test_loader, device=device, hidden_dim=HIDDEN_DIM, epochs=EPOCHS, lr=LR)
# ┌─ [BL-4] ─── run_bl_stgcn(train_loader, val_loader, test_loader, AP, device=device, hidden_dim=HIDDEN_DIM, epochs=EPOCHS, lr=LR)
# ┌─ [BL-5] ─── run_bl_dcrnn(train_loader, val_loader, test_loader, AP, device=device, hidden_dim=HIDDEN_DIM, epochs=EPOCHS, lr=LR)
# ┌─ [BL-6] ─── run_bl_gwn(train_loader, val_loader, test_loader, AP, device=device, hidden_dim=HIDDEN_DIM, epochs=EPOCHS, lr=LR)
# ┌─ [BL-7] ─── run_bl_astgcn(train_loader, val_loader, test_loader, AP, device=device, hidden_dim=HIDDEN_DIM, epochs=EPOCHS, lr=LR, T_in=H_RECENT)
#              ASTGCN: GPU 런타임으로 변경 후 실행 권장
#
# ┌─ [최종 비교 테이블] ────────────────────────────────────────────────────────
# │  results = load_bl_results()
# │
# │  # MCAST-ZINB 멀티스텝 결과 추가:
# │  #   본 노트북에서 evaluate_multistep() 으로 얻은 overall 지표를 넣고,
# │  #   horizon별 곡선까지 겹쳐 그리려면 'per_horizon' 리스트도 함께 넣는다.
# │  #   per_h, overall = evaluate_multistep(model, test_loader, SC, KC, AP, AA, AV)
# │  #   results['MCAST-ZINB+visual'] = {**overall, 'per_horizon': per_h}
# │
# │  # 전체 평균 비교 (기존)
# │  df = results_table(results)
# │
# │  # [멀티스텝] horizon별 표 + degradation curve (리뷰어 W2 대응)
# │  horizon_table(results, metric='mae')     # 모델 × h1..h24 표
# │  horizon_table(results, metric='hr_at_20')
# │  plot_degradation(results, metric='mae')  # x=horizon, y=MAE 곡선
# │  import matplotlib.pyplot as plt; plt.show()
# │
# │  # [멀티스텝] 논문용 wide 표 (1h/3h/6h × MAE/RMSE/Non-zero MAE/F1/HR@20%)
# │  order = ['HA','LSTM','GRU','STGCN','DCRNN','GraphWaveNet','ASTGCN',
# │           'MCAST-NB','MCAST-ZINB']
# │  df_wide = report_wide(results, horizons=(1, 3, 6), model_order=order)
# │  print(df_wide.to_string())
# │  save_report_wide(df_wide, 'horizon_report')   # csv/tex/xlsx
# │
# │  # [복잡도] 파라미터 수 + 추론 시간 (리뷰어 cost 대응)
# │  #   모든 모델을 같은 device·같은 배치로 측정해야 공정하다.
# │  batch = next(iter(test_loader))          # (x_r, x_d, x_w, temporal, y)
# │  cost = {}
# │  # baseline 예 (그래프 없음 / 있음):
# │  m_lstm = LSTMBaseline(HIDDEN_DIM, 2).to(device)      # 또는 학습된 모델 재사용
# │  cost['LSTM']  = measure_cost(m_lstm,
# │      build_forward_fn(m_lstm, batch, adj=None, device=device), device)
# │  m_stgcn = STGCNBaseline(HIDDEN_DIM).to(device)
# │  cost['STGCN'] = measure_cost(m_stgcn,
# │      build_forward_fn(m_stgcn, batch, adj=AP, device=device), device)
# │  # ... DCRNN/GWN/ASTGCN 도 adj=AP 로 동일하게
# │  # MCAST (forward 시그니처가 다름):
# │  cost['MCAST-ZINB'] = measure_cost(model,
# │      build_forward_fn(model, batch, device=device, mcast_args=dict(
# │          static_context=SC, keyword_context=KC,
# │          adj_phys=AP, adj_adapt=AA, adj_visual=AV)), device)
# │  cost_df = cost_table(cost, model_order=order)
# │  cost_df.to_csv('cost_report.csv')
# └────────────────────────────────────────────────────────────────────────────
