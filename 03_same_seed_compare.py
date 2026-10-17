"""03_same_seed_compare.py — 同种子公平对比: max-weight vs 比例分配

两个调度器面对**完全相同**的到达序列, 展示两个经典结论:

1. 稳定区 (默认, rho = 0.9): 两者都是 work-conserving (有能力就发),
   平均总队列完全相同 (Little 定律的体现); 差别只在**分布**——
   max-weight 把各终端队列压平, 比例分配则复现到达速率的不均衡;
2. 过载区 (--overload, rho = 1.01): 没有调度器能逃脱 rho > 1,
   但 max-weight 服务热点终端、压低最大队列, 比例分配会先崩溃。

运行:
    python 03_same_seed_compare.py             # 稳定场景
    python 03_same_seed_compare.py --overload  # 过载场景
    python 03_same_seed_compare.py --show
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===================== 系统参数 =====================
K      = 3      # 地面终端数量
T      = 1000   # 仿真总时隙
Q_MAX  = 500    # 单终端队列上限 (崩溃阈值, 包)
C_SAT  = 100    # 卫星总接收容量 (包/时隙)
PSI    = 10     # max-weight 权重下限 (包)

# 两个场景的平均到达速率 (包/时隙/终端)
R_STABLE  = np.array([40, 30, 20])   # Σ = 90  < C = 100  (rho = 0.90, 稳定)
R_OVERLOAD = np.array([81, 10, 10])  # Σ = 101 > C = 100  (rho = 1.01, 过载)

SEED = 42


def max_weight_scheduler(q_t, c_sat, psi=PSI):
    """Lyapunov max-weight: 按 (Q+Psi) 降序贪心装满容量"""
    weights = np.asarray(q_t) + psi
    v_t = np.zeros_like(q_t)
    remaining = c_sat
    for idx in np.argsort(-weights):
        if remaining <= 0:
            break
        send = min(q_t[idx], remaining)
        v_t[idx] = send
        remaining -= send
    return v_t


def proportional_scheduler(q_t, c_sat):
    """比例分配: 按队列长度比例分容量 (带整数微调), 空队列不分"""
    q_t = np.asarray(q_t)
    v_t = np.zeros_like(q_t)
    total_q = q_t.sum()
    if total_q == 0:
        return v_t

    for k in range(len(q_t)):
        v_t[k] = min(int(round(q_t[k] / total_q * c_sat)), q_t[k])

    # 四舍五入/截断造成的差额微调: 多补少减
    diff = c_sat - v_t.sum()
    if diff > 0:
        for k in np.argsort(-q_t):
            if q_t[k] > v_t[k] and diff > 0:
                add = min(diff, q_t[k] - v_t[k])
                v_t[k] += add
                diff -= add
    elif diff < 0:
        for k in np.argsort(-v_t):
            if v_t[k] > 0 and diff < 0:
                sub = min(-diff, v_t[k])
                v_t[k] -= sub
                diff += sub
    return v_t


def run_simulation(scheduler, r_all, c_sat=C_SAT, q_max=Q_MAX, **kwargs):
    """同一条到达序列跑一种调度器"""
    t_total, k = r_all.shape
    Q = np.zeros((t_total + 1, k), dtype=int)
    V = np.zeros((t_total, k), dtype=int)
    crash_time = -1
    for t in range(t_total):
        V[t] = scheduler(Q[t], c_sat, **kwargs)
        Q[t + 1] = np.maximum(Q[t] - V[t] + r_all[t], 0)
        if crash_time == -1 and np.any(Q[t + 1] > q_max):
            crash_time = t + 1
    return Q, V.sum(axis=1), crash_time


def main(r_means=None, save_dir='figures', show=False):
    if r_means is None:
        r_means = R_STABLE
    overload = r_means.sum() > C_SAT
    os.makedirs(save_dir, exist_ok=True)

    # 同一条泊松到达序列 (核心公平性保证)
    rng = np.random.default_rng(SEED)
    r_all = rng.poisson(r_means, (T, K))

    print(f'===== 03 同种子对比 ({"过载" if overload else "稳定"}场景) =====')
    print(f'R_means = {r_means.tolist()}, Σλ = {r_means.sum()}, '
          f'C = {C_SAT}, rho = {r_means.sum() / C_SAT:.2f}')

    results = {}
    for name, sched in [('MaxWeight', max_weight_scheduler),
                        ('Proportional', proportional_scheduler)]:
        Q, thr, crash = run_simulation(sched, r_all)
        results[name] = (Q, thr, crash)
        print(f'\n[{name}]  崩溃: {"是, 时隙 " + str(crash) if crash > 0 else "否"}')
        print(f'  最大队列: {Q.max():5d}   平均总队列: {Q.sum(axis=1).mean():7.2f}')
        print(f'  平均吞吐: {thr.mean():6.2f}  利用率: {thr.mean() / C_SAT * 100:5.1f}%')

    q1, _, c1 = results['MaxWeight']
    q2, _, c2 = results['Proportional']
    print('\n[结论] 平均总队列: '
          f'{q1.sum(axis=1).mean():.2f} vs {q2.sum(axis=1).mean():.2f} '
          '(work-conserving → 相同, Little 定律)')
    print(f'[结论] 各终端最大队列: {q1.max(axis=0).tolist()} vs {q2.max(axis=0).tolist()}')
    print(f'[结论] 崩溃时隙: {c1 if c1 > 0 else "无"} vs {c2 if c2 > 0 else "无"}')

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for row, (name, (Q, thr, crash)) in enumerate(results.items()):
        ax = axes[row, 0]
        for k in range(K):
            ax.plot(Q[:, k], label=f'terminal {k + 1}')
        ax.axhline(Q_MAX, color='r', ls='--', lw=1, label='crash threshold')
        ax.set_title(f'{name}: per-terminal queue'
                     + (f' (crash @{crash})' if crash > 0 else ''))
        ax.set_xlabel('slot')
        ax.set_ylabel('queue (packets)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[row, 1]
        ax.plot(thr, color='g', lw=0.8)
        ax.axhline(C_SAT, color='r', ls='--', lw=1)
        ax.set_title(f'{name}: throughput')
        ax.set_xlabel('slot')
        ax.grid(alpha=0.3)

        ax = axes[row, 2]
        ax.plot(Q.sum(axis=1), color='b')
        ax.axhline(K * Q_MAX, color='r', ls='--', lw=1)
        ax.set_title(f'{name}: total queue')
        ax.set_xlabel('slot')
        ax.grid(alpha=0.3)

    fig.suptitle(f'Same-seed comparison, rho = {r_means.sum() / C_SAT:.2f} '
                 f'({"overloaded" if overload else "stable"})')
    fig.tight_layout()
    tag = 'overload' if overload else 'stable'
    out = os.path.join(save_dir, f'03_same_seed_{tag}.png')
    fig.savefig(out, dpi=150)
    print(f'\n[OK] {out}')
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Same-seed max-weight vs proportional')
    parser.add_argument('--overload', action='store_true',
                        help='run the overloaded scenario (rho = 1.01)')
    parser.add_argument('--show', action='store_true', help='pop up plot windows')
    parser.add_argument('--save-dir', default='figures')
    args = parser.parse_args()
    main(r_means=R_OVERLOAD if args.overload else R_STABLE,
         save_dir=args.save_dir, show=args.show)
