"""
ablation.py — MCAST-ZINB Ablation Study
=========================================
사용법 (Colab):
    from google.colab import drive
    drive.mount('/content/drive')

    from ablation import run_ablation_study
    results = run_ablation_study(
        train_loader, val_loader, test_loader,
        SC, KC, AP, AA, AV,
        D_static=D_static, D_keyword=D_keyword, N=N,
        epochs=EPOCHS, lr=LR, device=device,
    )

체크포인트:
    - 5에폭마다 epoch_ckpt_{name}.pt 저장 (런타임 초기화 대비)
    - 실험 완료 시 best_{name}.pt + ablation_results.json 저장
    - 재실행 시 완료 실험은 스킵, 진행 중 실험은 마지막 에폭부터 재개
    - 기본 저장 경로: /content/drive/MyDrive/mcast_data/best model/Ablation/
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEFAULT_SAVE_DIR = './checkpoints/ablation'  # run_ablation_study(save_dir=...) 로 덮어쓸 수 있음
EPOCH_CKPT_INTERVAL = 5   # 몇 에폭마다 에폭 체크포인트 저장

# [멀티스텝] 예측 지평 — 본 노트북의 H_HORIZON 과 반드시 동일해야 함.
#   또한 threshold/evaluate 의 시점 정렬을 위해 H_RECENT/H_WEEKLY 도 맞춘다.
H_HORIZON = 24
H_RECENT  = 24
H_WEEKLY  = 24


# ════════════════════════════════════════════════════════════════
# 1. 실험 설정
# ════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    # (name,               use_sc, use_adapt, use_visual, use_kc, zinb)
    ('V1_temporal',        False,  False,     False,      False,  True ),
    ('V2_admin_db',        True,   True,      False,      False,  True ),
    ('V3_svi_only',        False,  False,     True,       True,   True ),
    ('V4_proposed',        True,   True,      True,       True,   True ),  # ★
    ('V5_wo_sc',           False,  True,      True,       True,   True ),
    ('V6_wo_adj_adapt',    True,   False,     True,       True,   True ),
    ('V7_visual_only',     True,   True,      True,       False,  True ),
    ('V8_kc_only',         True,   True,      False,      True,   True ),
    ('V9_nb',              True,   True,      True,       True,   False),
]
ABLATION_CONFIGS = [
    dict(name=n, use_sc=sc, use_adapt=ad,
         use_visual=vi, use_kc=kc, zinb=z)
    for n, sc, ad, vi, kc, z in ABLATION_CONFIGS
]


# ════════════════════════════════════════════════════════════════
# 2. 모델 구성요소
# ════════════════════════════════════════════════════════════════

def prepare_tcn_input(x, temporal, static_context, keyword_context):
    B, _, N, W = x.shape
    D_s = static_context.size(-1)
    D_k = keyword_context.size(-1)
    t_  = temporal.view(B, 4, 1, 1).expand(B, 4, N, W)
    s_  = static_context.T.unsqueeze(0).unsqueeze(-1).expand(B, D_s, N, W)
    k_  = keyword_context.T.unsqueeze(0).unsqueeze(-1).expand(B, D_k, N, W)
    return torch.cat([x, t_, s_, k_], dim=1)


class TemporalGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(dim * 3, 3), nn.Softmax(dim=-1))

    def forward(self, h_r, h_d, h_w):
        gates          = self.gate_layer(torch.cat([h_r, h_d, h_w], dim=-1))
        g_r, g_d, g_w = gates.unbind(dim=-1)
        return (g_r.unsqueeze(-1)*h_r +
                g_d.unsqueeze(-1)*h_d +
                g_w.unsqueeze(-1)*h_w)


class STBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dilation):
        super().__init__()
        _conv = lambda: nn.Conv2d(
            in_dim, out_dim,
            kernel_size=(1, 3), dilation=(1, dilation),
            padding=(0, dilation))
        self.tcn_r = _conv(); self.tcn_d = _conv(); self.tcn_w = _conv()
        self.fusion      = TemporalGatedFusion(out_dim)
        self.gcn_phys    = nn.Linear(out_dim, out_dim)
        self.gcn_adapt   = nn.Linear(out_dim, out_dim)
        self.gcn_visual  = nn.Linear(out_dim, out_dim)
        self.gcn_weights = nn.Parameter(torch.ones(3) / 3)
        self.norm        = nn.LayerNorm(out_dim)

    def forward(self, x_r, x_d, x_w, adj_phys, adj_adapt, adj_visual):
        h_r_seq = F.relu(self.tcn_r(x_r))
        h_d_seq = F.relu(self.tcn_d(x_d))
        h_w_seq = F.relu(self.tcn_w(x_w))

        h_r = h_r_seq[..., -1].permute(0, 2, 1)
        h_d = h_d_seq[..., -1].permute(0, 2, 1)
        h_w = h_w_seq[..., -1].permute(0, 2, 1)

        h_fused = self.fusion(h_r, h_d, h_w)
        B, N, D = h_fused.shape

        out_phys   = torch.matmul(adj_phys, self.gcn_phys(h_fused))
        gcn_in_ad  = self.gcn_adapt(h_fused)
        out_adapt  = torch.stack([
            torch.sparse.mm(adj_adapt, gcn_in_ad[b]) for b in range(B)])
        gcn_in_vi  = self.gcn_visual(h_fused)
        out_visual = torch.stack([
            torch.sparse.mm(adj_visual, gcn_in_vi[b]) for b in range(B)])

        w     = F.softmax(self.gcn_weights, dim=0)
        h_gcn = F.relu(self.norm(
            w[0]*out_phys + w[1]*out_adapt + w[2]*out_visual))

        h_seq     = (h_r_seq + h_d_seq + h_w_seq) / 3
        h_gcn_exp = h_gcn.permute(0, 2, 1).unsqueeze(-1).expand_as(h_seq)
        return h_seq + h_gcn_exp


class MCAST_ZINB(nn.Module):
    def __init__(self, static_dim, keyword_dim,
                 n_layers=4, hidden_dim=64,
                 use_zero_inflation=True):
        super().__init__()
        self.use_zero_inflation = use_zero_inflation
        first_in_dim = 1 + 4 + static_dim + keyword_dim
        self.blocks  = nn.ModuleList([
            STBlock(first_in_dim if i == 0 else hidden_dim,
                    hidden_dim, dilation=2**i)
            for i in range(n_layers)
        ])
        if use_zero_inflation:
            self.pi_head = nn.Linear(hidden_dim, H_HORIZON)   # [멀티스텝] 1 → H
        self.mu_head    = nn.Linear(hidden_dim, H_HORIZON)    # [멀티스텝]
        self.theta_head = nn.Linear(hidden_dim, H_HORIZON)    # [멀티스텝]

    def forward(self, x_r, x_d, x_w, temporal,
                static_context, keyword_context,
                adj_phys, adj_adapt, adj_visual):
        x_r_in = prepare_tcn_input(x_r, temporal, static_context, keyword_context)
        x_d_in = prepare_tcn_input(x_d, temporal, static_context, keyword_context)
        x_w_in = prepare_tcn_input(x_w, temporal, static_context, keyword_context)
        h_r, h_d, h_w = x_r_in, x_d_in, x_w_in
        for block in self.blocks:
            h = block(h_r, h_d, h_w, adj_phys, adj_adapt, adj_visual)
            h_r = h_d = h_w = h

        h_final = h[..., -1].permute(0, 2, 1)                 # (B, N, hidden)
        if self.use_zero_inflation:
            pi = torch.sigmoid(self.pi_head(h_final))         # [멀티스텝] squeeze 제거 → (B,N,H)
        else:
            # NB: pi=0 을 (B, N, H) 로 맞춘다   [멀티스텝]
            pi = torch.zeros(h_final.shape[0], h_final.shape[1], H_HORIZON,
                             device=h_final.device)
        mu    = F.softplus(self.mu_head(h_final))             # [멀티스텝] (B,N,H)
        theta = torch.exp(self.theta_head(h_final))           # [멀티스텝] (B,N,H)
        return pi, mu, theta


# ════════════════════════════════════════════════════════════════
# 3. Loss 함수
# ════════════════════════════════════════════════════════════════

def zinb_loss(y, pi, mu, theta, weight=None, eps=1e-8):
    nb_log_prob = (
        torch.lgamma(y+theta) - torch.lgamma(theta) - torch.lgamma(y+1)
        + theta*(torch.log(theta+eps) - torch.log(mu+theta+eps))
        + y    *(torch.log(mu+eps)    - torch.log(mu+theta+eps))
    )
    log_prob = torch.where(
        y < 1e-8,
        torch.log(pi + (1-pi)*torch.exp(nb_log_prob) + eps),
        torch.log(1-pi+eps) + nb_log_prob
    )
    if weight is not None:
        return -(log_prob * weight).sum() / (weight.sum() + eps)
    return -log_prob.mean()


def nb_loss(y, mu, theta, weight=None, eps=1e-8):
    log_prob = (
        torch.lgamma(y+theta) - torch.lgamma(theta) - torch.lgamma(y+1)
        + theta*(torch.log(theta+eps) - torch.log(mu+theta+eps))
        + y    *(torch.log(mu+eps)    - torch.log(mu+theta+eps))
    )
    if weight is not None:
        return -(log_prob * weight).sum() / (weight.sum() + eps)
    return -log_prob.mean()


def compute_loss(y, pi, mu, theta, weight=None, use_zinb=True):
    if use_zinb:
        return zinb_loss(y, pi, mu, theta, weight=weight)
    else:
        return nb_loss(y, mu, theta, weight=weight)


# ════════════════════════════════════════════════════════════════
# 4. 헬퍼 함수
# ════════════════════════════════════════════════════════════════

def make_zero_adj(N, device):
    idx = torch.zeros(2, 0, dtype=torch.long, device=device)
    val = torch.zeros(0, device=device)
    return torch.sparse_coo_tensor(idx, val, (N, N)).to_sparse_csr()


def get_ablation_tensors(cfg, SC, KC, AP, AA, AV, N, device):
    sc = SC if cfg['use_sc']     else torch.zeros_like(SC)
    kc = KC if cfg['use_kc']     else torch.zeros_like(KC)
    aa = AA if cfg['use_adapt']  else make_zero_adj(N, device)
    av = AV if cfg['use_visual'] else make_zero_adj(N, device)
    return sc, kc, AP, aa, av


def compute_extreme_threshold_per_node(train_loader, percentile=0.9, device='cuda'):
    _, _, _, _, target = next(iter(train_loader))
    N           = target.shape[1]
    node_values = [[] for _ in range(N)]
    for _, _, _, _, target in train_loader:
        y = target.to(device)
        for n in range(N):
            yn = y[:, n, :]                       # [멀티스텝] (B, H) — 모든 배치·horizon
            nz = yn[yn > 1e-8]
            if len(nz) > 0:
                node_values[n].append(nz.cpu())
    threshold = torch.zeros(N, device=device)
    for n in range(N):
        if not node_values[n]:
            threshold[n] = 1.0
        else:
            vals = torch.cat(node_values[n])
            threshold[n] = (torch.quantile(vals, percentile).to(device)
                            if len(vals) >= 10
                            else vals.max().to(device) * 0.5)
    return threshold


# ════════════════════════════════════════════════════════════════
# 5. 학습 / 평가
# ════════════════════════════════════════════════════════════════

def train_one_epoch(model, train_loader, optimizer,
                    static_context, keyword_context,
                    adj_phys, adj_adapt, adj_visual,
                    threshold_per_node=None, warmup=True,
                    use_zinb=True,
                    alpha=5.0, beta=2.0, entropy_reg=0.0001,
                    accumulation_steps=2, normalize_weight=True,
                    device='cuda'):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    eps = 1e-8

    for step, (x_r, x_d, x_w, temporal, target) in enumerate(train_loader):
        x_r, x_d, x_w = x_r.to(device), x_d.to(device), x_w.to(device)
        temporal, y    = temporal.to(device), target.to(device)

        pi, mu, theta = model(x_r, x_d, x_w, temporal,
                               static_context, keyword_context,
                               adj_phys, adj_adapt, adj_visual)

        if warmup:
            weight = torch.where(
                y > 1e-8, torch.ones_like(y), torch.full_like(y, 0.05))
        else:
            normalized     = torch.clamp(
                y / (threshold_per_node.view(1, -1, 1) + eps), 0, 2)  # [멀티스텝] (1,N,1)
            extreme_weight = 1.0 + beta * torch.relu(normalized - 1.0)
            if normalize_weight:
                extreme_weight /= (
                    extreme_weight.mean(dim=(0, 2), keepdim=True).detach() + eps)  # [멀티스텝] (0,2)
            weight = torch.where(
                y < eps, torch.ones_like(y), alpha * extreme_weight)

        loss = compute_loss(y, pi, mu, theta, weight=weight, use_zinb=use_zinb)

        if use_zinb and entropy_reg > 0:
            pi_entropy = -(pi*torch.log(pi+eps)
                           + (1-pi)*torch.log(1-pi+eps)).mean()
            loss = loss + entropy_reg * pi_entropy

        total_loss += loss.item()
        (loss / accumulation_steps).backward()

        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); optimizer.zero_grad()

    if len(train_loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step(); optimizer.zero_grad()

    return total_loss / len(train_loader)


def _compute_all_metrics(all_pi, all_mu, all_y, avg_loss=None):
    """
    [멀티스텝] (T,N) 또는 (T,N,H) 모두 처리. 원래 evaluate 의 지표를 모두 계산.
    HR@20 은 (시점,노드) 랭킹이라, (T,N,H)면 H를 시점축으로 펼쳐 (T*H, N)로 계산.
    """
    expected = (1 - all_pi) * all_mu
    if all_y.ndim == 3:                                  # (T, N, H)
        T, N, H = all_y.shape
        y_2d = np.transpose(all_y,     (0, 2, 1)).reshape(T * H, N)
        e_2d = np.transpose(expected,  (0, 2, 1)).reshape(T * H, N)
    else:                                                # (T, N)
        T, N = all_y.shape
        y_2d = all_y
        e_2d = expected

    y_flat        = y_2d.flatten()
    expected_flat = e_2d.flatten()

    zero_rate_true = float(np.mean(y_flat == 0))
    zero_rate_pred = float(np.mean(expected_flat < 0.5))

    y_sum = y_flat.sum(); exp_sum = expected_flat.sum()
    if y_sum > 0 and exp_sum > 0:
        p = y_flat / y_sum + 1e-10
        q = expected_flat / exp_sum + 1e-10
        kl_div = float(np.sum(p * np.log(p / q)))
    else:
        kl_div = float('nan')

    mae  = float(mean_absolute_error(y_flat, expected_flat))
    rmse = float(np.sqrt(mean_squared_error(y_flat, expected_flat)))

    pos_mask = y_flat > 0
    if pos_mask.sum() > 0:
        mae_pos  = float(mean_absolute_error(y_flat[pos_mask], expected_flat[pos_mask]))
        rmse_pos = float(np.sqrt(mean_squared_error(y_flat[pos_mask], expected_flat[pos_mask])))
    else:
        mae_pos = rmse_pos = float('nan')

    true_bin = (y_flat > 0).astype(int)
    pred_bin = (expected_flat > 0.5).astype(int)
    tp = int(((pred_bin == 1) & (true_bin == 1)).sum())
    fp = int(((pred_bin == 1) & (true_bin == 0)).sum())
    fn = int(((pred_bin == 0) & (true_bin == 1)).sum())
    recall    = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    n_slice = y_2d.shape[0]
    k = max(1, int(np.floor(0.2 * N)))
    hits = 0; events = 0
    for t in range(n_slice):
        true_nodes = np.where(y_2d[t] > 0)[0]
        if true_nodes.size == 0:
            continue
        events += len(true_nodes)
        top_k   = np.argsort(e_2d[t])[::-1][:k]
        hits   += np.intersect1d(top_k, true_nodes).size
    hr_at_20 = hits / events if events > 0 else float('nan')

    out = dict(
        zero_rate_true=zero_rate_true, zero_rate_pred=zero_rate_pred,
        kl_div=kl_div,
        mae=mae, rmse=rmse, mae_pos=mae_pos, rmse_pos=rmse_pos,
        recall_pos=float(recall), precision_pos=float(precision),
        f1_pos=float(f1), hr_at_20=hr_at_20,
    )
    if avg_loss is not None:
        out['loss'] = avg_loss
    return out


@torch.no_grad()
def evaluate(model, loader,
             static_context, keyword_context,
             adj_phys, adj_adapt, adj_visual,
             use_zinb=True, device='cuda'):
    """[멀티스텝] 전체 평균 지표. (B,N,H) 출력에서 동작. run_training(val)용."""
    model.eval()
    all_pi, all_mu, all_y = [], [], []
    total_loss = 0.0

    for x_r, x_d, x_w, temporal, target in loader:
        x_r, x_d, x_w = x_r.to(device), x_d.to(device), x_w.to(device)
        temporal, y    = temporal.to(device), target.to(device)
        pi, mu, theta  = model(x_r, x_d, x_w, temporal,
                                static_context, keyword_context,
                                adj_phys, adj_adapt, adj_visual)
        total_loss += compute_loss(y, pi, mu, theta, use_zinb=use_zinb).item()
        all_pi.append(pi.cpu()); all_mu.append(mu.cpu()); all_y.append(y.cpu())

    all_pi = torch.cat(all_pi).numpy()
    all_mu = torch.cat(all_mu).numpy()
    all_y  = torch.cat(all_y).numpy()
    return _compute_all_metrics(all_pi, all_mu, all_y, avg_loss=total_loss / len(loader))


@torch.no_grad()
def evaluate_multistep(model, loader,
                       static_context, keyword_context,
                       adj_phys, adj_adapt, adj_visual,
                       use_zinb=True, device='cuda'):
    """[멀티스텝] horizon(1..H)별 전체 지표 분해 + 전체 평균(overall) 반환."""
    model.eval()
    all_pi, all_mu, all_y = [], [], []
    total_loss = 0.0

    for x_r, x_d, x_w, temporal, target in loader:
        x_r, x_d, x_w = x_r.to(device), x_d.to(device), x_w.to(device)
        temporal, y    = temporal.to(device), target.to(device)
        pi, mu, theta  = model(x_r, x_d, x_w, temporal,
                                static_context, keyword_context,
                                adj_phys, adj_adapt, adj_visual)
        total_loss += compute_loss(y, pi, mu, theta, use_zinb=use_zinb).item()
        all_pi.append(pi.cpu()); all_mu.append(mu.cpu()); all_y.append(y.cpu())

    all_pi = torch.cat(all_pi).numpy()   # (T, N, H)
    all_mu = torch.cat(all_mu).numpy()
    all_y  = torch.cat(all_y).numpy()
    H = all_y.shape[-1]

    per_horizon = []
    for h in range(H):
        m = _compute_all_metrics(all_pi[:, :, h], all_mu[:, :, h], all_y[:, :, h])
        m['h'] = h + 1
        per_horizon.append(m)

    overall = _compute_all_metrics(all_pi, all_mu, all_y, avg_loss=total_loss / len(loader))
    return per_horizon, overall


def run_training(model, train_loader, val_loader,
                 static_context, keyword_context,
                 adj_phys, adj_adapt, adj_visual,
                 n_epochs=70, lr=5e-4, warmup_epochs=30,
                 use_zinb=True, device='cuda',
                 best_path='best_model.pt',
                 epoch_ckpt_path=None):
    """
    epoch_ckpt_path: 에폭 체크포인트 저장 경로
                     존재하면 해당 에폭부터 재개
    """
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=5, factor=0.5)
    threshold  = compute_extreme_threshold_per_node(
        train_loader, device=device)
    best_loss  = float('inf')
    start_epoch = 1

    # ── 에폭 체크포인트 로드 (런타임 재시작 시) ──────────────────
    if epoch_ckpt_path and Path(epoch_ckpt_path).exists():
        ckpt = torch.load(epoch_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        best_loss   = ckpt['best_loss']
        start_epoch = ckpt['epoch'] + 1
        print(f'  ↩️  에폭 체크포인트 로드: {ckpt["epoch"]}에폭까지 완료 → {start_epoch}에폭부터 재개')

    for epoch in range(start_epoch, n_epochs + 1):
        is_warmup  = (epoch <= warmup_epochs)
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            static_context, keyword_context,
            adj_phys, adj_adapt, adj_visual,
            threshold_per_node=threshold,
            warmup=is_warmup, use_zinb=use_zinb, device=device)
        val_m = evaluate(
            model, val_loader,
            static_context, keyword_context,
            adj_phys, adj_adapt, adj_visual,
            use_zinb=use_zinb, device=device)
        scheduler.step(val_m['loss'])

        mode = 'warmup' if is_warmup else 'full '
        print(f'[Epoch {epoch:3d}/{n_epochs} {mode}] '
              f'train={train_loss:.4f} | val_loss={val_m["loss"]:.4f} '
              f'mae={val_m["mae"]:.4f} f1={val_m["f1_pos"]:.4f}')

        # best 모델 저장
        if val_m['loss'] < best_loss:
            best_loss = val_m['loss']
            torch.save(model.state_dict(), best_path)
            print(f'  💾 Best saved (val_loss={best_loss:.4f}) → {best_path}')

        # ── 에폭 체크포인트 저장 (5에폭마다) ─────────────────────
        if epoch_ckpt_path and epoch % EPOCH_CKPT_INTERVAL == 0:
            torch.save({
                'epoch'          : epoch,
                'model_state'    : model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss'      : best_loss,
            }, epoch_ckpt_path)
            print(f'  📌 에폭 체크포인트 저장: {epoch}에폭 → {epoch_ckpt_path}')


# ════════════════════════════════════════════════════════════════
# 6. Ablation 전체 실행 (체크포인트 포함)
# ════════════════════════════════════════════════════════════════

def _get_paths(save_dir):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / 'ablation_results.json'
    return save_dir, results_path


def _load_progress(results_path):
    if Path(results_path).exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}


def _save_progress(results, results_path):
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


def _print_summary(results):
    metrics = ['loss', 'mae', 'rmse', 'mae_pos',
               'rmse_pos', 'f1_pos', 'hr_at_20']
    header = f'{"실험":<22}' + ''.join(f'{m:>10}' for m in metrics)
    print('\n' + '='*90)
    print('📊 Ablation Study 결과')
    print('='*90)
    print(header)
    print('-'*90)
    for name, m in results.items():
        row = f'{name:<22}' + ''.join(
            f'{m.get(k, float("nan")):>10.4f}' for k in metrics)
        print(row)
    print('='*90)


def run_ablation_study(
        train_loader, val_loader, test_loader,
        SC, KC, AP, AA, AV,
        D_static, D_keyword, N,
        epochs=70, lr=5e-4, warmup_epochs=30,
        device='cuda',
        save_dir=DEFAULT_SAVE_DIR,
        configs=None):
    if configs is None:
        configs = ABLATION_CONFIGS

    save_dir, results_path = _get_paths(save_dir)
    print(f'💾 저장 경로: {save_dir}')

    results = _load_progress(results_path)
    done    = set(results.keys())
    total   = len(configs)

    print(f'▶ Ablation Study 시작: 총 {total}개 실험')
    if done:
        print(f'  ✅ 이미 완료: {sorted(done)}')
        print(f'  ⏭  스킵 → 나머지 {total - len(done)}개만 실행')

    for idx, cfg in enumerate(configs, 1):
        name = cfg['name']

        if name in done:
            print(f'\n[{idx}/{total}] ⏭  {name} — 이미 완료, 스킵')
            continue

        print(f'\n{"="*65}')
        print(f'[{idx}/{total}] ▶ {name}')
        print(f'  SC={cfg["use_sc"]} | adj_adapt={cfg["use_adapt"]} | '
              f'adj_visual={cfg["use_visual"]} | KC={cfg["use_kc"]} | '
              f'ZINB={cfg["zinb"]}')
        print(f'{"="*65}')

        sc, kc, ap, aa, av = get_ablation_tensors(
            cfg, SC, KC, AP, AA, AV, N, device)

        model = MCAST_ZINB(
            static_dim=D_static,
            keyword_dim=D_keyword,
            use_zero_inflation=cfg['zinb'],
        ).to(device)
        print(f'  파라미터: {sum(p.numel() for p in model.parameters()):,}개')

        best_path      = str(save_dir / f'best_{name}.pt')
        epoch_ckpt_path = str(save_dir / f'epoch_ckpt_{name}.pt')

        run_training(
            model, train_loader, val_loader,
            sc, kc, ap, aa, av,
            n_epochs=epochs, lr=lr,
            warmup_epochs=warmup_epochs,
            use_zinb=cfg['zinb'],
            device=device,
            best_path=best_path,
            epoch_ckpt_path=epoch_ckpt_path,
        )

        # 테스트 평가
        model.load_state_dict(torch.load(best_path, map_location=device))
        per_h, test_m = evaluate_multistep(          # [멀티스텝] horizon별 + overall
            model, test_loader,
            sc, kc, ap, aa, av,
            use_zinb=cfg['zinb'], device=device)
        test_m['per_horizon'] = per_h                # 저장에 포함

        print(f'\n  ✅ Test | '
              f'loss={test_m["loss"]:.4f} | '
              f'mae={test_m["mae"]:.4f} | '
              f'mae_pos={test_m["mae_pos"]:.4f} | '
              f'f1={test_m["f1_pos"]:.4f} | '
              f'hr@20={test_m["hr_at_20"]:.4f}')

        print('  GCN 가중치 (수렴 후):')
        for i, block in enumerate(model.blocks):
            w = F.softmax(block.gcn_weights, dim=0).detach().cpu().numpy()
            print(f'    Block {i}: '
                  f'α(phys)={w[0]:.3f} | '
                  f'β(adapt)={w[1]:.3f} | '
                  f'γ(visual)={w[2]:.3f}')

        # 결과 저장
        results[name] = test_m
        _save_progress(results, results_path)
        print(f'  💾 결과 저장 → {results_path}')

        # 에폭 체크포인트 삭제 (실험 완료)
        if Path(epoch_ckpt_path).exists():
            Path(epoch_ckpt_path).unlink()
            print(f'  🗑  에폭 체크포인트 삭제 (실험 완료)')

    _print_summary(results)
    return results
