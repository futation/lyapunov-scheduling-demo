"""03 同种子对比测试: work-conserving 与稳定性结论"""
import numpy as np
import pytest

from conftest import load_module

m = load_module('03_same_seed_compare.py')


def test_proportional_work_conserving():
    """比例分配: 需求充足时用满容量, 需求不足时分完需求"""
    q = np.array([100, 200, 300])
    v = m.proportional_scheduler(q, c_sat=100)
    assert v.sum() == 100
    assert np.all(v <= q)

    q_small = np.array([5, 10, 15])
    v = m.proportional_scheduler(q_small, c_sat=100)
    assert v.sum() == 30
    assert (v == q_small).all()

    v0 = m.proportional_scheduler(np.zeros(3, dtype=int), c_sat=100)
    assert v0.sum() == 0


def test_max_weight_work_conserving():
    """max-weight 同样用满容量 (有积压时)"""
    q = np.array([100, 200, 300])
    v = m.max_weight_scheduler(q, c_sat=100)
    assert v.sum() == 100


def test_same_seed_identical_arrivals():
    """同种子生成完全相同的到达序列 (公平对比的前提)"""
    r1 = np.random.default_rng(m.SEED).poisson(m.R_STABLE, (m.T, m.K))
    r2 = np.random.default_rng(m.SEED).poisson(m.R_STABLE, (m.T, m.K))
    assert (r1 == r2).all()


def test_stable_scenario_survives():
    """rho = 0.9 稳定场景: 两种调度器都不崩, 平均总队列相同 (work conservation)"""
    rng = np.random.default_rng(m.SEED)
    r_all = rng.poisson(m.R_STABLE, (300, m.K))
    q1, _, c1 = m.run_simulation(m.max_weight_scheduler, r_all)
    q2, _, c2 = m.run_simulation(m.proportional_scheduler, r_all)
    assert c1 == -1 and c2 == -1
    assert q1.sum(axis=1).mean() == pytest.approx(q2.sum(axis=1).mean())


def test_overload_proportional_crashes_first():
    """rho = 1.01: 比例分配在本种子 T=1000 内崩溃且最大队列更大;
    max-weight 服务热点终端, 压低最大队列 (理论上渐近仍会崩, 无调度器能逃过过载)"""
    q_max_small = 500
    rng = np.random.default_rng(m.SEED)
    r_all = rng.poisson(m.R_OVERLOAD, (1000, m.K))
    q1, _, c1 = m.run_simulation(m.max_weight_scheduler, r_all, q_max=q_max_small)
    q2, _, c2 = m.run_simulation(m.proportional_scheduler, r_all, q_max=q_max_small)
    assert c2 > 0                       # 比例分配崩溃
    assert c1 == -1 or c1 > c2          # max-weight 撑得更久
    assert q1.max() <= q2.max()         # max-weight 最大队列更小
