import numpy as np
import matplotlib.pyplot as plt

trajectories = []
states = []

# 目标点和障碍物位置统一使用 numpy 向量
goal = np.array([10.0, 10.0], dtype=float)
pobs = np.array([5.0, 5.0], dtype=float)


def generate_trajectories(p):
    """从起点 p 生成一条到达 goal 的轨迹。"""
    # 确保 p 是 numpy 向量
    p = np.array(p, dtype=float)

    trajectory = []
    trajectory.append(p.copy())
    # 简单的终止条件：距离目标大于 1 时继续前进
    while np.linalg.norm(p - goal) > 1.0:
        # 吸引力：朝向目标 + 小随机扰动
        k_att = 1.0
        diff = p - goal
        noise = np.random.normal(0, 0.1, size=2)
        F_att = -k_att * diff + noise

        # 排斥势场参数
        d0 = 5.0
        offset = p - pobs
        dist = np.linalg.norm(offset)

        if dist < d0:
            dir_vec = offset / (dist + 1e-6)
            k_rep = 3.0
            F_rep = k_rep * (1.0 / dist - 1.0 / d0) / (dist**2 + 1e-6) * dir_vec
        else:
            F_rep = np.zeros(2, dtype=float)

        F_total = F_att + F_rep

        # 控制步长，避免一步跳得太大
        if np.linalg.norm(F_total) > 0:
            v = F_total / np.linalg.norm(F_total) * 0.1
        else:
            v = np.zeros(2, dtype=float)

        p = p + v
        trajectory.append(p.copy())

    return trajectory


def save_trajectories_txt(path, trajectories):
    """将 trajectories 写入 txt，每条轨迹一行。"""
    with open(path, "w") as f:
        for traj in trajectories:
            # 每个点写成 "x y"，点之间用分号分隔
            line_parts = []
            for point in traj:
                x, y = point
                line_parts.append(f"{x:.4f} {y:.4f}")
            line = "; ".join(line_parts)
            f.write(line + "\n")


def plot_sample_trajectories(trajectories, num_samples=10):
    """从 trajectories 中选取若干条轨迹进行绘制。"""
    num_samples = min(num_samples, len(trajectories))
    idx = np.random.choice(len(trajectories), size=num_samples, replace=False)

    plt.figure(figsize=(6, 6))
    for i in idx:
        traj = np.array(trajectories[i])
        plt.plot(traj[:, 0], traj[:, 1], alpha=0.7)

    # 画出目标点和障碍物
    plt.scatter(goal[0], goal[1], c="green", label="goal")
    circle = plt.Circle(pobs, 0.5, color="red", alpha=0.3)
    plt.gca().add_patch(circle)
    plt.scatter(pobs[0], pobs[1], c="red", label="obstacle center")

    plt.xlim(0, 12)
    plt.ylim(0, 12)
    plt.gca().set_aspect("equal", "box")
    plt.legend()
    plt.title("Sample Expert Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 生成 100 条轨迹
    for i in range(1000):
        p = np.random.uniform(0, 10, size=2)
        trajectory = generate_trajectories(p)
        trajectories.append(trajectory)

    # 保存到 txt，每条轨迹一行
    save_trajectories_txt("trajectories.txt", trajectories)

    # 随机选取 10 条轨迹并绘制
    plot_sample_trajectories(trajectories, num_samples=100)
