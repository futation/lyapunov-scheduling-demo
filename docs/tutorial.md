# Lyapunov 调度入门: 从漂移上界到 max-weight

本文是这个仓库的理论伴读: 用最小的篇幅讲清楚
"为什么 max-weight 调度能保证队列稳定", 并对应到三个示例脚本的代码。
公式风格参考 M. J. Neely, *Stochastic Network Optimization with
Application to Communication and Queueing Systems* (Morgan & Claypool, 2010)
——想看严格证明请直接读原书第 4 章。

---

## 1. 系统模型

K 个终端共享一条容量为 C 的上行链路 (卫星接收)。离散时隙 t = 0, 1, 2, ...

- 队列动态:

$$Q_k(t+1) = \max\big(Q_k(t) - V_k(t) + R_k(t),\ 0\big)$$

  其中 $R_k(t)$ 为随机到达, $V_k(t)$ 为调度器分配的服务量,
  约束 $\sum_k V_k(t) \le C$, $0 \le V_k(t) \le Q_k(t)$。

- **稳定**的定义: 队列长度的长期时间平均有界。
- **稳定域**: $\rho = \sum_k \lambda_k / C < 1$ ($\lambda_k$ 为平均到达率)。
  $\rho > 1$ 时**任何**调度器都无法稳定系统——这是 physically impossible,
  不是算法不够好 (对应 `03_same_seed_compare.py --overload`)。

## 2. 李雅普诺夫函数与漂移

把"队列积压总量"压成一个标量势函数:

$$L(t) = \frac{1}{2}\sum_k Q_k(t)^2$$

每时隙的漂移 $\Delta(t) = L(t+1) - L(t)$ 度量"积压是否在恶化"。
展开平方差并用 $a \le \max(a, 0)$ 放缩, 可得漂移上界:

$$\Delta(t) \le B + \sum_k Q_k(t)\,\big(\lambda_k - \mathbb{E}[V_k(t)]\big)$$

其中 $B$ 是由到达的二阶矩决定的常数 (与调度无关)。
**这个不等式是全部魔法的来源**: 右边唯一的可操作项是
$-\sum_k Q_k \mathbb{E}[V_k]$。

## 3. 漂移最小化 => max-weight

要让 $\Delta$ 尽量负 (队列被压下去), 每时隙解:

$$\max_V \ \sum_k Q_k(t)\, V_k(t) \quad \text{s.t. } \sum_k V_k \le C,\ V_k \le Q_k$$

这是一个线性规划, 闭式解: **把容量全部给队列最长的终端**
(`01_basic_max_weight.py` 的 `max_weight_scheduler`,
权重 $\Psi$ 只是防止空队列比较时的下限偏置)。

把这个规则代入漂移上界, 可以证明: 只要 $\rho < 1$,
时间平均队列有界 (max-weight 的"吞吐最优性")。
直观理解: 队列长度就是"漂移的价格信号", 服务最贵的队列,
漂移下降最快。

## 4. 加入物理层: 带宽分配

`02_channel_bandwidth.py` 把抽象的服务量 $V_k$ 换成香农速率:

$$V_k = W_k \log_2\!\Big(1 + \frac{p_k h_k}{N_0 W_k}\Big)$$

决策变量从"服务量"变成"带宽切分" $\{W_k\}$, 目标不变:

$$\max_{\{W_k\}} \ \sum_k (Q_k + \Psi)\, W_k \log_2\Big(1 + \frac{p_k h_k}{N_0 W_k}\Big)
\quad \text{s.t. } \sum_k W_k \le W_{total}$$

目标对 $W$ 非凸 (log 里带 W), 用 SLSQP 数值求解——这就是
"Lyapunov 漂移 + 每时隙一个凸/非凸子问题"的通用骨架:
**主问题(跨时隙)被 Lyapunov 拆解成一串单时隙优化**。

两个工程陷阱 (本仓库实测踩过):

1. **N0 量纲**: 热噪声是 -174 dBm/Hz ≈ 4e-21 W/Hz。
   若随手写 1e-6, 速率全部趋零, 调度器全崩, 对比无意义。
2. **Ψ 尺度**: $\sum(Q+\Psi)V = \sum Q V + \Psi\sum V$,
   Ψ 过大时后一项主导, 调度器退化成"吞吐最大化", 弱信道被饿。

## 5. work conservation 与 Little 定律

`03_same_seed_compare.py` 对比 max-weight 与"按队列比例分配":

- 两者都是 **work-conserving** (有积压就发满), 因此总队列的时间平均相同
  (实测 90.86 vs 90.86);
- 差别只在**分布**: max-weight 压平各队列, 比例分配复现到达不均衡;
- $\rho > 1$ 时两者都会崩, 但 max-weight 撑得更久、最大队列更小
  (服务热点终端)。

结论: **平均性能往往由"是否偷懒"决定 (work-conserving 就够了),
调度器的真正价值体现在尾部 (最大队列/崩溃时间)。**
评估调度器时, 别只看平均值。

## 6. 与真实论文的关系

真实系统 (LEO 卫星 IoT 接入控制等) 在此骨架上增加: 更复杂的信道
(LoS/NLoS、衰落)、能量约束 (再引入虚拟队列 $Z(t)$)、DRL 求解器
(把 max-weight 换成策略网络)。姊妹仓库 `lyapunov-ppo-uav-mec`
展示了"虚拟队列进观测 + RL 学习调度"的完整例子。
骨架不变: 队列动态 → 势函数 → 漂移上界 → 逐时隙决策。
