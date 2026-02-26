"""
扩散模型推理：从纯噪声 A_T 出发，用训练好的 epsilon 网络逐步去噪，得到动作序列 A_0。
使用 DDPM 反向公式：给定 A_t 和预测的 eps，算出 A_{t-1}。
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from noise import NoiseScheduler
from epsilon_net import ConditionalEpsilonNet, Horizon, Action_dim


def sample_actions(model, scheduler, S_t, num_samples=1, device="cpu", T=100):
    """
    从纯噪声采样得到 A_0。
    S_t: [B, 2] 或 [2,]（当前状态，条件）
    返回 A_0: [B, 16, 2]
    """
    model.eval()
    with torch.no_grad():
        if S_t.dim() == 1:
            S_t = S_t.unsqueeze(0)
        S_t = S_t.to(device)
        B = S_t.shape[0]

        # 从纯噪声开始，形状与动作序列一致
        A_t = torch.randn(B, Horizon, Action_dim, device=device)

        # 从 t=T 逐步去噪到 t=1
        for t in range(T, 0, -1):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            eps_pred = model(A_t, t_batch, S_t)

            # DDPM 反向步：用 alpha_t, alpha_bar_t, beta_t 从 A_t 得到 A_{t-1}
            idx = t - 1  # 0-based
            alpha_t = scheduler.alphas[idx]
            alpha_bar_t = scheduler.alpha_cumprod[idx]
            beta_t = scheduler.betas[idx]

            # 均值：DDPM 公式 mu_t = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps)
            coef_eps = beta_t / torch.sqrt(1 - alpha_bar_t)
            mean = (1.0 / torch.sqrt(alpha_t)) * (A_t - coef_eps * eps_pred)

            if t > 1:
                # 方差：DDPM 公式，需要 alpha_bar_{t-1}
                alpha_bar_prev = scheduler.alpha_cumprod[idx - 1]
                var_t = (1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t
                sigma_t = torch.sqrt(var_t)
                noise = torch.randn_like(A_t, device=device)
                A_t = mean + sigma_t * noise
            else:
                A_t = mean  # t=1 时直接取均值作为 A_0

        return A_t


def rollout_until_goal(
    model, scheduler, S_start, goal,
    goal_threshold=1.0, max_iters=50, device="cpu", T=100,
    steps_per_sample=2,
):
    """
    从起点 S_start 开始，反复「采样 16 个动作 → 只执行前 steps_per_sample 步 → 用新 S 再预测」直到到达 goal。
    steps_per_sample=2：每次只取前 2 个预测动作执行并更新 S，再重新采样，形成闭环反馈。
    """
    if isinstance(S_start, np.ndarray):
        s = S_start.copy()
    else:
        s = S_start.cpu().numpy().flatten()
    goal = np.asarray(goal, dtype=float)
    traj = [s.copy()]
    k_execute = min(max(1, steps_per_sample), Horizon)

    for _ in range(max_iters):
        if np.linalg.norm(s - goal) <= goal_threshold:
            break
        S_t = torch.from_numpy(s).float().unsqueeze(0).to(device)  # [1, 2]
        A_0 = sample_actions(model, scheduler, S_t, num_samples=1, device=device, T=T)
        A_0 = A_0.cpu().numpy()[0]  # (16, 2)
        for k in range(k_execute):
            s = s + A_0[k]
            traj.append(s.copy())
            if np.linalg.norm(s - goal) <= goal_threshold:
                return np.array(traj)
    return np.array(traj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="epsilon_net.pt")
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--state", type=str, default=None, help="起点坐标，如 '0,0' 或 '1,2'；不传则随机")
    parser.add_argument("--out", type=str, default="sampled_actions.png")
    parser.add_argument("--rollout", action="store_true", default=True, help="闭环：未到 goal 附近则继续采样行走（默认开）")
    parser.add_argument("--no_rollout", action="store_false", dest="rollout", help="只采样一段 16 步，不闭环")
    parser.add_argument("--goal_threshold", type=float, default=1.0)
    parser.add_argument("--max_iters", type=int, default=100)
    parser.add_argument("--steps_per_sample", type=int, default=1, help="每次采样 16 个动作后只执行前 k 步再重新预测，默认 2")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps":
        try:
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                device = "cpu"
        except Exception:
            device = "cpu"

    scheduler = NoiseScheduler(T=args.T, beta_start=1e-4, beta_end=0.02, device=device)
    model = ConditionalEpsilonNet(T=args.T, cond_dim=64).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    goal = np.array([10.0, 10.0])
    if args.state:
        parts = args.state.split(",")
        starts = [np.array([float(parts[0]), float(parts[1])])]
    else:
        starts = [np.random.rand(2) * 2 for _ in range(args.num_samples)]

    if args.rollout:
        # 闭环：从每个起点走到 goal 附近为止
        trajectories = []
        for s0 in starts:
            traj = rollout_until_goal(
                model, scheduler, s0, goal,
                goal_threshold=args.goal_threshold,
                max_iters=args.max_iters,
                device=device, T=args.T,
                steps_per_sample=args.steps_per_sample,
            )
            trajectories.append(traj)
        S_np = np.array(starts)
    else:
        # 原逻辑：只采样一段 A_0，转成轨迹
        S_t = torch.tensor(np.array(starts), dtype=torch.float32, device=device)
        A_0 = sample_actions(model, scheduler, S_t, num_samples=S_t.shape[0], device=device, T=args.T)
        A_0 = A_0.cpu().numpy()
        S_np = S_t.cpu().numpy()
        trajectories = []
        for i in range(A_0.shape[0]):
            s = S_np[i].copy()
            traj = [s.copy()]
            for k in range(A_0.shape[1]):
                s = s + A_0[i, k]
                traj.append(s.copy())
            trajectories.append(np.array(traj))

    # 画图：多条采样轨迹 + 目标 (10,10) + 障碍 (5,5)
    obs = np.array([5.0, 5.0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.7)
    ax.scatter(S_np[:, 0], S_np[:, 1], c="blue", s=50, label="start")
    ax.scatter(goal[0], goal[1], c="green", s=80, marker="*", label="goal")
    circle = plt.Circle(obs, 0.5, color="red", alpha=0.3)
    ax.add_patch(circle)
    ax.scatter(obs[0], obs[1], c="red", s=30, label="obstacle")
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.set_aspect("equal", "box")
    ax.legend()
    ax.set_title("Sampled Trajectories (from diffusion)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    n_traj = len(trajectories)
    total_pts = sum(len(t) for t in trajectories)
    mode = "rollout (until goal)" if args.rollout else "single horizon"
    print(f"Sampled {n_traj} trajectories ({mode}), total points {total_pts}, plot saved to {args.out}")


if __name__ == "__main__":
    main()
