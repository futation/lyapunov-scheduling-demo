# lyapunov-scheduling-demo

**用 ~100 行代码讲透李雅普诺夫优化。** 三个可独立运行的小脚本, 从抽象的
共享下行队列讲到真实香农速率下的带宽分配, 回答一个问题:
*为什么 max-weight 调度能保证队列稳定*。

[English](README.md) · [理论伴读 (中文)](docs/tutorial.md)

| ![](docs/img/01_basic_max_weight.png) | ![](docs/img/02_channel_bandwidth.png) |
|---|---|
| ![](docs/img/03_same_seed_stable.png) | ![](docs/img/03_same_seed_overload.png) |

## 三个脚本各讲一件事

1. **`01_basic_max_weight.py`** — 抽象队列 + 闭式调度。
   李雅普诺夫函数 `L(t) = ½·ΣQ²`、漂移上界 `Δ ≤ B + ΣQ·(λ−V)`,
   最小化漂移 ⟹ "谁队列长先服务谁"。50 行看懂核心。
2. **`02_channel_bandwidth.py`** — 加入物理层。
   服务量换成香农速率, 每时隙用 SLSQP 解带宽切分。
   场景故意让"固定均分带宽"失配 → 四种调度器只有 Lyapunov 活下来;
   顺带展示"吞吐最优 ≠ 队列稳定"(max-throughput 第 58 时隙就崩)。
3. **`03_same_seed_compare.py`** — 同种子公平对比。
   稳定区: work-conserving 调度器平均总队列完全相同 (Little 定律),
   差别只在分布; 过载区: 没人能逃过 ρ>1, 但 max-weight 撑得更久。

## 实测结果 (脚本随仓库参数, 种子固定)

**02 — 负载/分配错配**: 终端 3 到达 95 kbit/slot, 故意超过均分带宽下的
服务速率 87.2:

| 调度器 | 结果 |
|---|---|
| **Lyapunov (max-weight)** | **存活**, 最大队列 ≈ 464 kbit |
| 固定均分 | 第 446 时隙崩溃 |
| 轮询 | 第 284 时隙崩溃 |
| 纯吞吐量最大化 | 第 58 时隙崩溃 |

**03 — 同种子, ρ = 0.9 与 ρ = 1.01**:

- 稳定区: 两者平均总队列完全相同 (90.86); 分布不同——
  max-weight 压平 (`[61, 58, 61]`), 比例分配复现到达不均衡 (`[63, 51, 35]`);
- 过载区: 谁也逃不掉, 但 max-weight 最大队列 450 vs 848,
  比例分配第 452 时隙先崩。

## 快速上手

```bash
pip install -r requirements.txt

python 01_basic_max_weight.py
python 02_channel_bandwidth.py          # SLSQP 逐时隙求解, 约 10 秒
python 03_same_seed_compare.py          # 稳定场景
python 03_same_seed_compare.py --overload

python -m pytest tests/ -q              # 17 个测试
```

图保存到 `figures/`, 加 `--show` 弹窗显示。值得玩的旋钮 (各脚本头部):
`RHO`/`R_MEAN` (推过 1 看全崩)、`PSI` (调太大 max-weight 会退化成
吞吐最大化, 见 02 的 Ψ 说明)、`Q_MAX`、`SEED`。

## 理论伴读

[docs/tutorial.md](docs/tutorial.md): 漂移上界推导、max-weight 归约、
Ψ 与 N₀ 两个工程陷阱, 以及这套骨架如何长成真实系统
(能量约束加虚拟队列 Z(t)、max-weight 换成 DRL)。

## 出处

理论框架遵循 M. J. Neely, *Stochastic Network Optimization with
Application to Communication and Queueing Systems* (Morgan & Claypool, 2010)。
本练习写于研读 LEO 卫星 IoT 接入调度文献期间
(参见 *"Joint Optimization of Access Control and Resource Allocation for LEO
Satellite IoT Using Deep Reinforcement Learning"*, IEEE TCCN 2026)——
本仓库只重实现其 Lyapunov 内核用于教学, 不复现论文模型。

## 许可

MIT, 见 [LICENSE](LICENSE)。
