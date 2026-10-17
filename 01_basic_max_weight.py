"""01_basic_max_weight.py — 李雅普诺夫最大权重调度, 50 行看懂核心思想

最抽象的入门模型: 3 个终端共享 1 份卫星接收容量, 队列以"包"计,
调度器每时隙决定容量怎么分。没有任何信道/速率细节——
"哪个队列长就先服务谁"这一件事, 就是 Lyapunov 漂移最小化的闭式解。

数学对应 (推导见 docs/tutorial.md):
    Lyapunov 函数  L(t) = 1/2 * Σ_k Q_k(t)^2
    漂移上界       Δ(t) ≤ B + Σ_k Q_k * E[R_k - V_k]
    最小化漂移     =>   每时隙最大化  Σ_k Q_k * V_k   (max-weight)
    在约束 Σ V_k <= C, 0 <= V_k <= Q_k 下的最优解:
       按 (Q_k + Ψ) 从大到小贪心装满容量。

运行:
    python 01_basic_max_weight.py            # 保存图到 figures/
    python 01_basic_max_weight.py --show     # 同时弹窗显示
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
PSI    = 10     # Lyapunov 权重下限 (防止空队列饿死, 量纲=包)
RHO    = 0.80   # 目标负载系数 rho = 总到达/总容量 (0.80 = 稳定区; 试 1.05 看崩溃)
R_MEAN = RHO * C_SAT / K   # 各终端平均到达速率 (包/时隙), 均分负载

SEED = 42


def max_weight_scheduler(q_t, c_sat, psi=PSI):
    """Lyapunov max-weight 闭式调度: 按 (Q+Psi) 降序贪心分配容量

    约束: sum(V) <= c_sat, 0 <= V_k <= Q_k
    """
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


def run_simulation(T=T, K=K, q_max=Q_MAX, c_sat=C_SAT, r_mean=R_MEAN,
                   seed=SEED, scheduler=max_weight_scheduler, **kwargs):
    """泊松到达 + 调度 + 队列更新, 返回轨迹与崩溃信息"""
    rng = np.random.default_rng(seed)
    R_all = rng.poisson(r_mean, (T, K))

    Q = np.zeros((T + 1, K), dtype=int)
    V = np.zeros((T, K), dtype=int)
    crash_time = -1
    for t in range(T):
        V[t] = scheduler(Q[t], c_sat, **kwargs)
        Q[t + 1] = np.maximum(Q[t] - V[t] + R_all[t], 0)
        if crash_time == -1 and np.any(Q[t + 1] > q_max):
            crash_time = t + 1
    throughput = V.sum(axis=1)
    return Q, V, throughput, crash_time


def main(save_dir='figures', show=False):
    os.makedirs(save_dir, exist_ok=True)
    Q, V, thr, crash = run_simulation()

    print('===== 01 基础 max-weight 调度 =====')
    print(f'负载系数 rho = {RHO:.2f} (总到达 {R_MEAN * K:.1f} / 容量 {C_SAT})')
    print(f'系统崩溃: {"是" if crash > 0 else "否"}'
          + (f', 崩溃时隙 {crash}' if crash > 0 else ''))
    print(f'最大队列: {Q.max()}  平均队列: {Q.mean():.2f}')
    print(f'平均吞吐: {thr.mean():.2f} 包/时隙  '
          f'容量利用率: {thr.mean() / C_SAT * 100:.1f}%')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in range(K):
        axes[0].plot(Q[:, k], label=f'terminal {k + 1}')
    axes[0].axhline(Q_MAX, color='r', ls='--', lw=1, label=f'crash threshold {Q_MAX}')
    axes[0].set(xlabel='slot t', ylabel='queue length (packets)',
                title=f'Max-weight scheduler (rho={RHO:.2f})')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(thr, color='g', lw=0.8)
    axes[1].axhline(C_SAT, color='r', ls='--', lw=1, label='capacity')
    axes[1].set(xlabel='slot t', ylabel='throughput (packets/slot)',
                title='Total service per slot')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(save_dir, '01_basic_max_weight.png')
    fig.savefig(out, dpi=150)
    print(f'[OK] {out}')
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic Lyapunov max-weight demo')
    parser.add_argument('--show', action='store_true', help='pop up plot windows')
    parser.add_argument('--save-dir', default='figures')
    args = parser.parse_args()
    main(save_dir=args.save_dir, show=args.show)
