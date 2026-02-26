import torch
import numpy as np
import matplotlib.pyplot as plt

class NoiseScheduler:
    def __init__(self, T, beta_start, beta_end,device = 'mps'):
        self.T = T
       
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x0, t, eps=None):
        """加噪：x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*eps。t 可为标量或 (B,)；返回 (x_t, eps)。"""
        x0 = x0.to(self.device)
        if eps is None:
            eps = torch.randn_like(x0, device=self.device)
        else:
            eps = eps.to(self.device)
        # t 为 1-based；支持标量（可视化）或 (B,)（训练）
        if isinstance(t, int) or (isinstance(t, torch.Tensor) and t.dim() == 0):
            idx = int(t) - 1
            idx = max(0, min(idx, self.T - 1))
            sqrt_abar = torch.sqrt(self.alpha_cumprod[idx])
            sqrt_1_abar = torch.sqrt(1.0 - self.alpha_cumprod[idx])
        else:
            B = x0.shape[0]
            idx = (t - 1).clamp(0, self.T - 1)
            sqrt_abar = torch.sqrt(self.alpha_cumprod[idx]).view(B, 1, 1)
            sqrt_1_abar = torch.sqrt(1.0 - self.alpha_cumprod[idx]).view(B, 1, 1)
        xt = sqrt_abar * x0 + sqrt_1_abar * eps
        return xt, eps

def load_data(path):
    with open(path, 'r') as f:
        line = f.readline().strip()
    
    points = []
    for item in line.split(';'):
        x, y = item.split()
        points.append([float(x), float(y)])
    x0 = torch.tensor(points, dtype=torch.float32)  # (L,2)
    return x0

def visualize_noising_process(x0, scheduler, ts=(1, 10, 30, 60, 100)):
    # 固定随机种子，方便复现
    torch.manual_seed(42)

    fig, axes = plt.subplots(1, len(ts), figsize=(4 * len(ts), 4))
    if len(ts) == 1:
        axes = [axes]

    for ax, t in zip(axes, ts):
        xt, _ = scheduler.add_noise(x0, t)

        x_np = xt[:, 0].cpu().numpy()
        y_np = xt[:, 1].cpu().numpy()

        # 用点显示更容易看噪声化
        ax.scatter(x_np, y_np, s=10, alpha=0.8)
        ax.plot(x_np, y_np, alpha=0.4)  # 可选：连线看形状扭曲
        ax.set_title(f"t={t}")
        ax.set_aspect("equal", "box")

    plt.tight_layout()
    plt.show()  

if __name__ == "__main__":
    x0 = load_data("trajectories.txt")
    scheduler = NoiseScheduler(T=100, beta_start=0.0001, beta_end=0.02, device="mps")
    visualize_noising_process(x0, scheduler)
