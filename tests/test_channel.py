"""02 信道带宽分配测试: 速率模型 + 调度器可行性 + 队列稳定对比"""
import numpy as np
import pytest

from conftest import load_module

m = load_module('02_channel_bandwidth.py')


def test_rate_zero_when_no_bandwidth():
    """带宽为 0 时速率为 0 (无除零告警/NaN)"""
    r = m.achievable_rate(np.array([0.0, 100.0, 5000.0]), m.H_MEAN)
    assert r[0] == 0.0
    assert np.isfinite(r).all()


def test_rate_monotonic_in_bandwidth():
    """速率随分配带宽单调不减"""
    h = m.H_MEAN
    rates = [m.achievable_rate(np.full(3, w), h) for w in (100, 1000, 5000, 10000)]
    for a, b in zip(rates, rates[1:]):
        assert np.all(a <= b)


def test_realistic_noise_floor():
    """N0 必须是真实热噪声量级 (-174 dBm/Hz), 防止回归到虚高量纲"""
    assert m.N0 == pytest.approx(10 ** ((-174 - 30) / 10), rel=1e-9)
    # 均分带宽下速率应为可用量级 (kbit/slot), 而不是趋近 0
    cap = m.achievable_rate(np.full(3, m.W_TOTAL / 3), m.H_MEAN)
    assert cap.min() > 50.0


def test_lyapunov_allocation_feasible():
    """Lyapunov 调度返回的带宽分配满足 sum(W) <= W_total, W >= 0"""
    rng = np.random.default_rng(3)
    for _ in range(5):
        q = rng.uniform(0, 1000, size=3)
        h = rng.exponential(m.H_MEAN)
        _, w = m.lyapunov_bandwidth_scheduler(q, h)
        assert w.sum() <= m.W_TOTAL + 1e-3
        assert np.all(w >= -1e-6)


def test_service_capped_by_queue():
    """服务量不超过队列长度 (所有调度器)"""
    q = np.array([10.0, 500.0, 50.0])
    h = m.H_MEAN
    for sched in (m.lyapunov_bandwidth_scheduler, m.fixed_bandwidth_scheduler,
                  m.max_throughput_scheduler):
        v, _ = sched(q, h)
        assert np.all(v <= q + 1e-9)


def test_lyapunov_survives_fixed_crashes():
    """负载-分配错配场景 (终端 3 到达 > 其均分服务):
    固定分配崩溃, Lyapunov 存活 — 本仓库核心教学结论的回归测试"""
    t_total = 200
    lam = np.array([90.0, 60.0, 115.0])       # 115 > 均分服务 87.2
    r_all = np.full((t_total, 3), lam.reshape(1, 3))   # 确定性到达
    h_all = np.full((t_total, 3), m.H_MEAN)

    q_fix, _, crash_fix = m.run_simulation(
        lambda q, h, w, t: m.fixed_bandwidth_scheduler(q, h, w),
        t_total=t_total, r_all=r_all, h_all=h_all)
    q_lya, _, crash_lya = m.run_simulation(
        lambda q, h, w, t: m.lyapunov_bandwidth_scheduler(q, h, w),
        t_total=t_total, r_all=r_all, h_all=h_all)

    assert crash_fix > 0, "fixed 应在失配负载下崩溃"
    assert crash_lya == -1, "Lyapunov 应存活"
    assert q_lya.max() < m.Q_MAX


def test_max_throughput_starves_weak_channel():
    """纯吞吐量最大化让弱信道终端队列失控 (吞吐最优 != 队列稳定)"""
    t_total = 120
    lam = np.array([90.0, 60.0, 115.0])
    r_all = np.full((t_total, 3), lam.reshape(1, 3))
    h_all = np.full((t_total, 3), m.H_MEAN)
    q, _, crash = m.run_simulation(
        lambda q, h, w, t: m.max_throughput_scheduler(q, h, w),
        t_total=t_total, r_all=r_all, h_all=h_all)
    assert q[:, 2].mean() > 10 * q[:, 0].mean()
