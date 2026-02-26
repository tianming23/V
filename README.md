# 扩散模型轨迹生成 (Diffusion for Action Sequence)

基于 DDPM 的条件扩散模型：在 2D 平面上，给定当前状态 \(S_t\)，生成「从当前点朝向目标 (10,10)、并绕过障碍 (5,5)」的未来动作序列 \(A_0\)，用于轨迹控制。

---

## 环境

- Python 3.8+
- PyTorch（建议支持 MPS/CUDA 以加速训练）
- NumPy, Matplotlib

```bash
pip install torch numpy matplotlib
```

---

## 项目结构

| 文件 | 说明 |
|------|------|
| `datageneration.py` | 专家数据生成：人工势场法从随机起点到 (10,10)，绕过 (5,5) 障碍，轨迹写入 `trajectories.txt` |
| `cuttra.py` | 轨迹切片：将长轨迹切成 (S_t, A_{t:t+H})，H=16，保存为 `expert_dataset_h16.npz` |
| `noise.py` | 加噪调度器 NoiseScheduler（线性 β），前向加噪公式与加噪可视化 |
| `epsilon_net.py` | 条件 1D-CNN：输入 A_t, t, S_t，预测噪声 ε；主干为 Conv1d + Mish + GroupNorm + 残差，条件为 t 与 S_t 相加后 FiLM 注入 |
| `train.py` | 训练：MSE(ε_pred, ε)，支持 MPS/CUDA/CPU，可保存 loss 曲线 |
| `sample.py` | 推理：从纯噪声采样 A_0，支持「每次只执行前 k 步再重新预测」的闭环 rollout |

---

## 流程概览

```
1. 生成专家轨迹     →  trajectories.txt
2. 切片为 (S,A)     →  expert_dataset_h16.npz
3. 训练 ε 网络      →  epsilon_net.pt, loss_curve.png
4. 采样 / 闭环 rollout →  sampled_actions.png
```

---

## 前向加噪公式（Noise Schedule）

扩散步数 \(t = 1, \ldots, T\)，\(\beta_t\) 线性调度（如从 \(10^{-4}\) 到 \(0.02\)），定义：

\[
\alpha_t = 1 - \beta_t, \qquad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s.
\]

**前向一步加噪**（从干净动作 \(A_0\) 得到带噪 \(A_t\)，\(\varepsilon \sim \mathcal{N}(0, I)\)）：

\[
A_t = \sqrt{\bar{\alpha}_t}\, A_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon.
\]

训练时对每个样本随机采样 \(t\) 和 \(\varepsilon\)，用网络预测 \(\varepsilon\)，损失为 \(\mathcal{L} = \mathbb{E}\big[\|\varepsilon - \varepsilon_\theta(A_t, t, S_t)\|^2\big]\)（MSE）。

---

## DDPM 反向公式（去噪采样）

从 \(A_T \sim \mathcal{N}(0, I)\) 出发，按 \(t = T, T-1, \ldots, 1\) 逐步去噪。每步用网络预测 \(\hat{\varepsilon} = \varepsilon_\theta(A_t, t, S_t)\)，则 \(A_{t-1}\) 的**后验均值**为：

\[
\mu_t = \frac{1}{\sqrt{\alpha_t}} \left( A_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\, \hat{\varepsilon} \right).
\]

**方差**（DDPM 取值）：

\[
\sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\, \beta_t.
\]

**采样**：
- 若 \(t > 1\)：\(A_{t-1} = \mu_t + \sigma_t\, z\)，其中 \(z \sim \mathcal{N}(0, I)\)；
- 若 \(t = 1\)：\(A_0 = \mu_1\)（不再加噪）。

代码见 `noise.py`（前向）、`sample.py`（反向）。

---

## 使用说明

### 1. 生成专家轨迹

在 `diffusion1` 目录下运行，生成多条轨迹并写入 `trajectories.txt`，并可选画 10 条示例图：

```bash
cd diffusion1
python datageneration.py
```

- 目标点：(10, 10)  
- 障碍：圆心 (5, 5)，排斥半径由势场参数控制  
- 输出：`trajectories.txt`（每行一条轨迹，点为 `x y; x y; ...`）

### 2. 切片为训练数据

将轨迹切成长度 H=16 的「当前状态 + 未来 16 步动作」样本，动作定义为相邻状态差：

```bash
python cuttra.py
```

- 输入：`trajectories.txt`（若路径不同，需在 `cuttra.py` 中修改）  
- 输出：`expert_dataset_h16.npz`，包含 `states` (N, 2)、`actions` (N, 16, 2)  
- 可在 `cuttra.py` 中修改 `horizon` 与保存路径

### 3. 训练扩散模型

用专家数据训练「预测前向噪声 ε」的网络：

```bash
python train.py --data expert_dataset_h16.npz --epochs 50 --batch_size 64 --lr 1e-4 --save epsilon_net.pt --plot loss_curve.png
```

- 默认会尝试使用 MPS（Mac GPU）或 CUDA；无则用 CPU  
- 输出：`epsilon_net.pt`（权重）、`loss_curve.png`（训练 loss 曲线）

### 4. 采样与闭环 rollout

从训练好的模型采样动作序列，并可按「每次只执行前 k 步再重新预测」的方式闭环到目标附近：

```bash
# 默认：每次采样 16 个动作，只执行前 2 步，再根据新 S 重新预测
python sample.py --ckpt epsilon_net.pt --out sampled_actions.png

# 指定起点
python sample.py --state 0,0

# 每次执行前 4 步再重新预测
python sample.py --steps_per_sample 4

# 每次执行整段 16 步（旧行为）
python sample.py --steps_per_sample 16

# 增加最大迭代次数以便走得更远
python sample.py --steps_per_sample 2 --max_iters 200
```

- 输出：`sampled_actions.png`，图中为多条从起点到目标附近的轨迹，以及障碍 (5,5) 与目标 (10,10)。

---

## 主要超参数

| 位置 | 参数 | 含义 |
|------|------|------|
| 扩散 | T=100, β 线性 1e-4→0.02 | 扩散步数与噪声调度 |
| 数据 | horizon=16 | 每段动作序列长度 |
| 训练 | batch_size=64, lr=1e-4 | 批大小与学习率 |
| 采样 | steps_per_sample=2, max_iters=50 | 每次执行步数、最大 rollout 迭代次数 |

---

## 引用与参考

- DDPM: Denoising Diffusion Probabilistic Models (Ho et al.)
- 条件注入：时间步 t 与状态 S_t 编码后相加，经 FiLM 注入 1D-CNN 主干
