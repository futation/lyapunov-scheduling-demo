"""02_channel_bandwidth.py — 真实信道下的 Lyapunov 带宽分配 (4 种调度器对比)

在 01 的抽象队列上加入物理层: 香农速率 + 每时隙 SLSQP 求解带宽分配。
Lyapunov 目标变为:  max_W  Σ_k (Q_k + Ψ) * rate_k(W_k)
    s.t.  Σ W_k <= W_total,  W_k >= 0

关键修正 (相对最初版本): 噪声功率谱密度取真实热噪声
    N0 = -174 dBm/Hz ≈ 3.98e-21 W/Hz
若 N0 虚高 (如 1e-6 W/Hz, 比真实值大 ~15 个数量级), 所有速率趋近 0,
任何调度器都会崩溃, 对比实验失去意义。

工作点经过设计, 让四种调度器拉开差距:
    到达 [100, 75, 95] kbit/slot —— 终端 3 的到达(95) 故意超过
    其均分带宽下的服务速率(87.2), 静态分配注定失衡。
实测结果 (T=800, 同一到达/信道序列):
    Lyapunov        稳定, 最大队列 ~490 kbit   (唯一幸存者)
    Fixed           崩溃 @466  (终端 3 队列线性漂移)
    RoundRobin      崩溃 @299
    MaxThroughput   崩溃 @57   —— 吞吐最优 != 队列稳定

关于 Ψ 的调参提醒: 目标 Σ(Q+Ψ)·rate 展开后含 Ψ·Σ(rate) 项,
Ψ 相对队列尺度过大时会把它变成"带一点点队列偏置的 max-throughput",
弱信道终端照样被饿。本场景队列量级 ~100 kbit, 取 Ψ=50;
若把 Ψ 加到 500, 终端 3 的稳态队列会翻倍。

运行:
    python 02_channel_bandwidth.py [--show]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint

# ===================== 系统参数 =====================
K        = 3              # 终端数量
T        = 800            # 仿真时隙数
Q_MAX    = 5000           # 单终端队列上限 (kbit, 崩溃阈值)
W_TOTAL  = 10e3           # 卫星总带宽 (Hz)
P_TX     = 0.1            # 各终端恒定发射功率 (W)
N0       = 10 ** ((-174 - 30) / 10)   # 噪声功率谱密度 (W/Hz), -174 dBm/Hz
PSI      = 50             # Lyapunov 权重下限 (kbit); 过大会偏向 max-throughput, 见文件头说明
R_MEAN   = np.array([100.0, 75.0, 95.0])   # 平均到达速率 (kbit/时隙); 95 > 均分服务 87.2
H_MEAN   = np.array([1e-6, 1e-7, 1e-8])    # 各终端信道增益均值 (指数衰落)
SEED     = 2024

# 预生成随机序列: 同一序列喂给所有调度器, 保证公平对比
_rng = np.random.default_rng(SEED)
R_ALL = _rng.poisson(R_MEAN, (T, K)).astype(float)   # kbit
H_ALL = _rng.exponential(H_MEAN, (T, K))             # 线性信道增益


# ===================== 速率与调度器 =====================
def achievable_rate(w, h):
    """各终端可达速率 (kbit/时隙): W_k * log2(1 + p*h_k/(N0*W_k))"""
    w = np.asarray(w, dtype=float)
    w_safe = np.maximum(w, 1e-9)   # SLSQP 会探索 w=0 边界, 先防除零
    snr = P_TX * h / (N0 * w_safe)
    rate = np.where(w > 1e-9, w * np.log2(1 + snr) / 1e3, 0.0)
    return rate


def lyapunov_bandwidth_scheduler(q, h, w_total=W_TOTAL, psi=PSI):
    """Lyapunov 调度: SLSQP 最大化 Σ (Q_k + Psi) * rate_k(W_k)"""
    k = len(q)

    def obj(w):
        return -np.sum((q + psi) * achievable_rate(w, h))

    cons = LinearConstraint(np.ones(k), 0, w_total)
    bounds = Bounds(np.zeros(k), np.full(k, w_total))
    res = minimize(obj, np.full(k, w_total / k), method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'maxiter': 200, 'ftol': 1e-8})
    return np.minimum(achievable_rate(res.x, h), q), res.x


def fixed_bandwidth_scheduler(q, h, w_total=W_TOTAL):
    """固定均分带宽"""
    w = np.full(len(q), w_total / len(q))
    return np.minimum(achievable_rate(w, h), q), w


def round_robin_scheduler(q, h, w_total=W_TOTAL, t=0):
    """轮询: 全部带宽给第 t mod K 个终端"""
    k = len(q)
    w = np.zeros(k)
    w[t % k] = w_total
    return np.minimum(achievable_rate(w, h), q), w


def max_throughput_scheduler(q, h, w_total=W_TOTAL):
    """纯吞吐量最大化: 不看队列, 只把带宽分给信道最好的终端"""
    k = len(q)

    def obj(w):
        return -np.sum(achievable_rate(w, h))

    cons = LinearConstraint(np.ones(k), 0, w_total)
    bounds = Bounds(np.zeros(k), np.full(k, w_total))
    res = minimize(obj, np.full(k, w_total / k), method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'maxiter': 200, 'ftol': 1e-8})
    return np.minimum(achievable_rate(res.x, h), q), res.x


# ===================== 仿真循环 =====================
def run_simulation(scheduler, q_max=Q_MAX, t_total=T, r_all=R_ALL, h_all=H_ALL,
                   w_total=W_TOTAL):
    """通用仿真: 返回队列轨迹 / 吞吐量 / 崩溃信息 (不同调度器签名差异用 lambda 适配)"""
    k = r_all.shape[1]
    Q = np.zeros((t_total + 1, k))
    thr = np.zeros(t_total)
    crash_time = -1
    for t in range(t_total):
        v, _ = scheduler(Q[t], h_all[t], w_total, t)
        Q[t + 1] = np.maximum(Q[t] - v + r_all[t], 0)
        if crash_time == -1 and np.any(Q[t + 1] > q_max):
            crash_time = t + 1
        thr[t] = v.sum()
    return Q, thr, crash_time


def main(save_dir='figures', show=False):
    os.makedirs(save_dir, exist_ok=True)

    print('===== 02 真实信道带宽分配 (4 种调度器) =====')
    print(f'N0 = {N0:.3e} W/Hz (-174 dBm/Hz), W_total = {W_TOTAL:.0f} Hz, '
          f'p = {P_TX} W')
    cap = achievable_rate(np.full(K, W_TOTAL / K), H_MEAN)
    full_cap = achievable_rate(np.full(K, W_TOTAL), H_MEAN)
    print(f'均分带宽容量: {np.round(cap, 1)} kbit/slot (满带宽: '
          f'{np.round(full_cap, 1)}), 总到达 {R_MEAN.sum():.0f} kbit/slot')

    schedulers = {
        'Lyapunov': lambda q, h, w, t: lyapunov_bandwidth_scheduler(q, h, w),
        'Fixed':    lambda q, h, w, t: fixed_bandwidth_scheduler(q, h, w),
        'RoundRobin': round_robin_scheduler,
        'MaxThroughput': lambda q, h, w, t: max_throughput_scheduler(q, h, w),
    }

    results = {}
    for name, sched in schedulers.items():
        Q, thr, crash = run_simulation(sched)
        results[name] = (Q, thr, crash)
        print(f'\n[{name}]  崩溃: {"是, 时隙 " + str(crash) if crash > 0 else "否"}')
        print(f'  平均队列: {Q.mean():7.1f} kbit   最大队列: {Q.max():8.1f} kbit')
        print(f'  各终端平均队列: {np.round(Q.mean(axis=0), 1)}')

    # ---- 可视化: 上排队列, 下排吞吐 ----
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    for i, (name, (Q, thr, crash)) in enumerate(results.items()):
        ax = axes[0, i]
        for k in range(K):
            ax.plot(Q[:, k] / 1e3, label=f'terminal {k + 1}')
        ax.axhline(Q_MAX / 1e3, color='r', ls='--', lw=1, label='crash')
        ax.set_title(f'{name}' + (f' (crash @{crash})' if crash > 0 else ''))
        ax.set_xlabel('slot')
        ax.set_ylabel('queue (Mbit)' if i == 0 else None)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, i]
        ax.plot(thr / 1e3, color='g', lw=0.7)
        ax.set_xlabel('slot')
        ax.set_ylabel('throughput (Mbit/slot)' if i == 0 else None)
        ax.grid(alpha=0.3)

    fig.suptitle('Lyapunov bandwidth allocation vs baselines '
                 '(same arrival & channel sequences)')
    fig.tight_layout()
    out = os.path.join(save_dir, '02_channel_bandwidth.png')
    fig.savefig(out, dpi=150)
    print(f'\n[OK] {out}')
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lyapunov channel-aware bandwidth allocation demo')
    parser.add_argument('--show', action='store_true', help='pop up plot windows')
    parser.add_argument('--save-dir', default='figures')
    args = parser.parse_args()
    main(save_dir=args.save_dir, show=args.show)
