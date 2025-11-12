"""
SDE Solver (随机微分方程求解器) 算法实现。

基于 Song et al. 2021 的 Score-based Generative Modeling，使用 SDE 框架建模扩散过程。
支持 VP-SDE (Variance Preserving) 类型的随机微分方程求解。

参考文献:
    Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
    Score-based generative modeling through stochastic differential equations.
"""

from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class SDESolver(nn.Module):
    """
    基于随机微分方程（SDE）的扩散模型求解器。

    实现 VP-SDE（Variance Preserving SDE）前向和反向过程。
    使用 Euler-Maruyama 方法进行数值求解。

    Args:
        n_timesteps (int): 离散化时间步数。
        beta_start (float): 初始噪声水平。
        beta_end (float): 最终噪声水平。
        sde_type (str, optional): SDE 类型，当前仅支持 'vpsde'。默认为 'vpsde'。
        device (str, optional): 计算设备。默认为 'cuda'。
    """

    def __init__(
        self,
        n_timesteps: int,
        beta_start: float,
        beta_end: float,
        sde_type: str = 'vpsde',
        device: str = 'cuda'
    ) -> None:
        super().__init__()

        if sde_type != 'vpsde':
            raise ValueError("当前仅支持 VP-SDE，请使用 sde_type='vpsde'")

        self.n_timesteps = n_timesteps
        self.sde_type = sde_type
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 线性 beta 调度
        self.beta = torch.linspace(beta_start, beta_end, n_timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # SDE 系数
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def forward_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向加噪过程（与 DDPM 相同）。"""
        noise = torch.randn_like(x_0)
        sqrt_alpha_bar_t = self._extract(self.sqrt_alpha_bar, t, x_0.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bar, t, x_0.shape
        )
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    def reverse_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """单步反向去噪（SDE 反向过程）。"""
        # 使用模型预测 score（等价于噪声预测）
        predicted_noise = model(x_t, t, condition)

        # 提取系数
        beta_t = self._extract(self.beta, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alpha_bar, t, x_t.shape
        )

        # Score: ∇log p(x_t) ≈ -ε / √(1-ᾱ_t)
        score = -predicted_noise / sqrt_one_minus_alpha_bar_t

        # VP-SDE 反向漂移项: f_rev = -0.5 * beta_t * x_t - beta_t * score
        # 由于反向时间，更新时使用 -f_rev
        # -f_rev = 0.5 * beta_t * x_t + beta_t * score
        drift = 0.5 * beta_t * x_t + beta_t * score

        # SDE 扩散项: g(t) = sqrt(beta_t)
        diffusion = torch.sqrt(beta_t) * torch.randn_like(x_t)

        # Euler-Maruyama 时间步长
        dt = 1.0 / self.n_timesteps

        # 反向 SDE 更新: x_{t-dt} = x_t + drift * dt + diffusion * sqrt(dt)
        x_t_minus_1 = x_t + drift * dt + diffusion * np.sqrt(dt)

        return x_t_minus_1

    def reverse_sample_loop(
        self,
        x_T: torch.Tensor,
        condition: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """完整反向采样循环。"""
        x_t = x_T

        for time_step in reversed(range(self.n_timesteps)):
            t = torch.full(
                (x_t.shape[0],), time_step, dtype=torch.long, device=self.device
            )
            x_t = self.reverse_sample(x_t, t, condition, model)

        return x_t

    def _extract(
        self, coefficients: torch.Tensor, t: torch.Tensor, shape: torch.Size
    ) -> torch.Tensor:
        """提取系数并广播。"""
        batch_size = t.shape[0]
        extracted = coefficients.gather(-1, t)
        return extracted.view(batch_size, *([1] * (len(shape) - 1)))


# 测试模块
if __name__ == "__main__":
    import sys, io, os
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.unet import SimpleUNet

    print("=" * 70)
    print("SDE Solver 算法测试")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}\n")

    sde = SDESolver(n_timesteps=100, beta_start=0.0001, beta_end=0.02, device=str(device))
    model = SimpleUNet(1, 1).to(device)

    x_0 = torch.randn(4, 1, 28, 28).to(device)
    t = torch.randint(0, 100, (4,)).to(device)
    x_t, noise = sde.forward_sample(x_0, t)

    print(f"✅ 前向加噪测试通过！输出形状: {x_t.shape}")

    condition = torch.randn(4, 1, 28, 28).to(device)
    with torch.no_grad():
        x_t_minus_1 = sde.reverse_sample(x_t, t, condition, model)
        print(f"✅ 单步反向去噪测试通过！输出形状: {x_t_minus_1.shape}")

        x_T = torch.randn(4, 1, 28, 28).to(device)
        x_0_gen = sde.reverse_sample_loop(x_T, condition, model)
        print(f"✅ 完整采样循环测试通过！输出形状: {x_0_gen.shape}")

    print("\n" + "=" * 70)
    print("SDE Solver 算法所有测试通过！✅")
    print("=" * 70)
