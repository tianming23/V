import numpy as np


def load_trajectories(path):
    """从 trajectories.txt 读取轨迹，返回 list[list[np.ndarray(2,)]]。"""
    trajectories = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            points_str = line.split(";")
            traj = []
            for p_str in points_str:
                p_str = p_str.strip()
                if not p_str:
                    continue
                x_str, y_str = p_str.split()
                traj.append(np.array([float(x_str), float(y_str)], dtype=float))
            if traj:
                trajectories.append(traj)
    return trajectories


def traj_to_samples(trajectories, horizon=16):
    """
    将长轨迹切成样本：
    输入：S_t（当前 state）
    标签：A_{t:t+H}（未来 H 步的动作序列），动作 = S_{k+1} - S_k
    """
    states = []
    actions_seq = []

    for traj in trajectories:
        traj = np.array(traj)  # shape: (T, 2)
        T = len(traj)
        # 需要至少 H+1 个点才能得到 H 个动作
        max_t = T - horizon - 1
        if max_t < 0:
            continue
        for t in range(max_t + 1):
            S_t = traj[t]  # (2,)
            # 未来 H 步对应的 H 个动作：S_{k+1} - S_k
            future_states = traj[t + 1 : t + 1 + horizon]  # (H, 2)
            current_states = traj[t : t + horizon]         # (H, 2)
            A_seq = future_states - current_states         # (H, 2)

            states.append(S_t)
            actions_seq.append(A_seq)

    states = np.array(states)              # (N, 2)
    actions_seq = np.array(actions_seq)    # (N, H, 2)
    return states, actions_seq


if __name__ == "__main__":
    trajs = load_trajectories("trajectories.txt")
    states, action_seqs = traj_to_samples(trajs, horizon=16)
    print("states shape:", states.shape)
    print("action_seqs shape:", action_seqs.shape)
    np.savez("expert_dataset_h3.npz", states=states, actions=action_seqs)
