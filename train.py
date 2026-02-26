"""
扩散模型训练：用专家数据 (S_t, A_0) 训练 epsilon 网络，loss = MSE(eps_pred, eps)。
数据需为 horizon=16：states (N,2), actions (N,16,2)。
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from noise import NoiseScheduler
from epsilon_net import ConditionalEpsilonNet, Horizon, Action_dim


class ExpertDataset(Dataset):
    def __init__(self, npz_path, device="cpu"):
        data = np.load(npz_path)
        self.states = torch.from_numpy(data["states"]).float()   # (N, 2)
        self.actions = torch.from_numpy(data["actions"]).float() # (N, H, 2)
        assert self.actions.shape[1] == Horizon, "需要 horizon=16，请用 cuttra 生成 H=16 的数据"
        self.device = device

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        return self.states[i], self.actions[i]


def train_step(model, batch, scheduler, device, T):
    S_t, A_0 = batch
    S_t = S_t.to(device)
    A_0 = A_0.to(device)
    B = S_t.shape[0]

    # 每个样本随机一个时间步 t，1-based
    t = torch.randint(1, T + 1, (B,), device=device, dtype=torch.long)

    # 前向加噪（用 noise.py 的 add_noise，返回 A_t 和 eps 便于监督）
    A_t, eps = scheduler.add_noise(A_0, t)

    eps_pred = model(A_t, t, S_t)
    loss = nn.functional.mse_loss(eps_pred, eps)
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="expert_dataset_h16.npz", help="npz: states, actions (N,16,2)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--save", type=str, default="epsilon_net.pt")
    parser.add_argument("--plot", type=str, default="loss_curve.png", help="保存 loss 曲线图路径，空则不画图")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps":
        try:
            mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
            if not mps_ok:
                device = "cpu"
                print("当前 Python 下 MPS 不可用，已用 CPU。若要用 Mac GPU，请用「能跑出 MPS: True」的同一解释器，例如：")
                print("  python3 train.py ...  或  /Users/hh/miniforge3/bin/python train.py ...")
        except Exception:
            device = "cpu"
            print("MPS 检测异常，已用 CPU。")

    print(f"Training device: {device}")
    dataset = ExpertDataset(args.data, device=device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    scheduler = NoiseScheduler(T=args.T, beta_start=1e-4, beta_end=0.02, device=device)
    model = ConditionalEpsilonNet(T=args.T, cond_dim=64).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loss_history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            loss = train_step(model, batch, scheduler, device, args.T)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        loss_history.append(avg)
        print(f"epoch {epoch+1}/{args.epochs}  loss {avg:.6f}")

    torch.save({"model": model.state_dict()}, args.save)
    print(f"Saved to {args.save}")

    if args.plot:
        plt.figure(figsize=(6, 4))
        plt.plot(loss_history, color="steelblue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        plt.close()
        print(f"Loss curve saved to {args.plot}")


if __name__ == "__main__":
    main()
