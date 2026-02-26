import torch
import torch.nn as nn
import math

Horizon = 16
Action_dim = 2

def sinusoidal_embedding(t, dim):
    """输入 t: [B] 整数；输出: [B, dim] 这和按列合并是强相关的""" 
    half_dim = dim // 2
    freqs = torch.exp(torch.arange(half_dim, device=t.device) * -(math.log(10000.0) / half_dim))
    t_float = t.float().unsqueeze(1)  # [B, 1]
    args = t_float * freqs * 2 * math.pi
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding


class ResBlock1d(nn.Module):
    """Conv1d + Mish + GroupNorm，带残差；通道变化时用 1x1 卷积对齐 shortcut。"""
    def __init__(self, in_ch, out_ch, num_groups=8):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.act = nn.Mish()
        self.shortcut = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)

    def forward(self, x):
        h = self.conv(x)
        h = self.act(h)
        h = self.norm(h)
        return h + self.shortcut(x)


class ConditionalEpsilonNet(nn.Module):
    def __init__(self, T =100,t_emb_dim = 128,s_emb_dim = 64,cond_dim = 128,channels = (32,64,64)):
       super().__init__()
       self.T = T
       self.cond_dim = cond_dim
       self.channels = (Action_dim,) + tuple(channels)

       # 正弦编码输出 [B, t_emb_dim]，所以第一层输入应为 t_emb_dim
       self.t_proj = nn.Sequential(
           nn.Linear(t_emb_dim, t_emb_dim),
           nn.SiLU(),
           nn.Linear(t_emb_dim, cond_dim),
       )
       self.s_proj = nn.Sequential(
           nn.Linear(Action_dim, s_emb_dim),
           nn.SiLU(),
           nn.Linear(s_emb_dim, cond_dim),
       )
       # 主干：stem + 3 层带残差的 Conv1d + Mish + GroupNorm（小 batch 用 GroupNorm 更稳）
       self.stem = nn.Conv1d(Action_dim, channels[0], kernel_size=3, padding=1)  # 2 -> 32
       self.blocks = nn.ModuleList([
           ResBlock1d(channels[0], channels[1]),   # 32 -> 64
           ResBlock1d(channels[1], channels[2]),   # 64 -> 64
           ResBlock1d(channels[2], channels[2]),   # 64 -> 64
       ])
       self.out_conv = nn.Conv1d(channels[-1], Action_dim, kernel_size=3, padding=1)
       # 每层主干后注入 cond：stem 输出 32，三个 block 输出 64,64,64
       backbone_channels = [channels[0]] + [channels[1], channels[2], channels[2]]
       self.cond_to_channel = nn.ModuleList([
           nn.Linear(cond_dim, c) for c in backbone_channels
       ])
    
    def forward(self, A_t, t, S_t):
        # A_t: [B, 2, 16]
        # t: [B]
        # S_t: [B, 2]
        B = A_t.shape[0]
        # 条件：t 与 S_t 同维后相加 -> cond_feature (B, cond_dim)，基础 FiLM
        t_emb = sinusoidal_embedding(t, self.t_proj[0].in_features)
        t_cond = self.t_proj(t_emb)   # [B, cond_dim]
        s_cond = self.s_proj(S_t)    # [B, cond_dim]
        cond_feature = t_cond + s_cond   # [B, cond_dim]，例如 (B, 64)

        x = A_t.transpose(1, 2)   # [B, 2, 16]

        # 主干：stem -> FiLM -> 3 个 ResBlock，每层后 FiLM 注入
        x = self.stem(x)
        x = x + self.cond_to_channel[0](cond_feature).unsqueeze(-1)

        for i, block in enumerate(self.blocks):
            x = block(x)
            x = x + self.cond_to_channel[i + 1](cond_feature).unsqueeze(-1)

        x = self.out_conv(x)      # [B, 2, 16]
        return x.transpose(1, 2)  # [B, 16, 2]