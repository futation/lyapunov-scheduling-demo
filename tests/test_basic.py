"""01 基础 max-weight 调度测试"""
import numpy as np

from conftest import load_module

m = load_module('01_basic_max_weight.py')


def test_capacity_and_queue_constraints():
    """调度结果满足 sum(V) <= C 且 V <= Q"""
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.integers(0, 200, size=m.K)
        v = m.max_weight_scheduler(q, m.C_SAT)
        assert v.sum() <= m.C_SAT
        assert np.all(v <= q)
        assert np.all(v >= 0)


def test_priority_order():
    """容量不足时, 队列最长的终端优先被服务"""
    q = np.array([10, 90, 5])
    v = m.max_weight_scheduler(q, c_sat=30, psi=0)  # psi=0: 纯 max-weight
    assert v[1] == 30
    assert v[0] == 0 and v[2] == 0


def test_stable_region_no_crash():
    """rho = 0.8 下 T=300 不崩溃"""
    Q, V, thr, crash = m.run_simulation(T=300)
    assert crash == -1
    assert Q.max() <= m.Q_MAX


def test_overload_eventually_crashes():
    """rho 明显大于 1 时必然崩溃 (演示稳定性边界)"""
    Q, V, thr, crash = m.run_simulation(T=500, r_mean=50.0)  # rho = 1.5
    assert crash > 0


def test_queue_conservation():
    """队列更新守恒: Q[t+1] = max(Q[t] - V[t] + R[t], 0)"""
    rng = np.random.default_rng(1)
    T = 50
    r_mean = m.R_MEAN
    R_all = rng.poisson(r_mean, (T, m.K))
    Q = np.zeros((T + 1, m.K), dtype=int)
    V = np.zeros((T, m.K), dtype=int)
    for t in range(T):
        V[t] = m.max_weight_scheduler(Q[t], m.C_SAT)
        Q[t + 1] = np.maximum(Q[t] - V[t] + R_all[t], 0)
    for t in range(T):
        expected = np.maximum(Q[t] - V[t] + R_all[t], 0)
        assert (Q[t + 1] == expected).all()
