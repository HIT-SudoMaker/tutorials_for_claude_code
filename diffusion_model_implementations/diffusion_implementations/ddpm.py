"""
DDPM (Denoising Diffusion Probabilistic Models) 算法实现。

基于 Ho et al. 2020 的 DDPM 论文，实现经典的去噪扩散概率模型。
该算法通过 Markov 链逐步添加高斯噪声，训练模型逐步去噪以生成数据。

参考文献:
    Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models.
    Advances in Neural Information Processing Systems, 33, 6840-6851.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import math


class DDPM(nn.Module):
    """
    DDPM 扩散模型实现。

    实现了前向加噪过程和反向去噪过程，支持配置线性或余弦噪声调度。
    所有核心参数（beta、alpha、alpha_bar 等）在初始化时预计算并移至指定设备。

    Args:
        n_timesteps (int): 总扩散时间步数 T（通常为 1000）。
        beta_start (float): 初始噪声水平 β_0（通常为 0.0001）。
        beta_end (float): 最终噪声水平 β_T（通常为 0.02）。
        beta_schedule (str, optional): Beta 调度方案。
            可选 'linear'（线性调度）或 'cosine'（余弦调度）。
            默认为 'linear'。
        device (str, optional): 计算设备（'cuda' 或 'cpu'）。默认为 'cuda'。

    Attributes:
        n_timesteps (int): 总时间步数。
        beta (torch.Tensor): Beta 序列，shape (T,)。
        alpha (torch.Tensor): Alpha 序列 (1 - beta)，shape (T,)。
        alpha_bar (torch.Tensor): Alpha 累积乘积序列，shape (T,)。
        sqrt_alpha_bar (torch.Tensor): √alpha_bar，用于前向过程。
        sqrt_one_minus_alpha_bar (torch.Tensor): √(1 - alpha_bar)，用于前向过程。
        sqrt_recip_alpha (torch.Tensor): 1/√alpha，用于反向过程。
        beta_tilde (torch.Tensor): 后验方差，用于反向过程。

    Example:
        >>> ddpm = DDPM(n_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        >>> x_0 = torch.randn(4, 1, 28, 28)
        >>> t = torch.randint(0, 1000, (4,))
        >>> x_t, noise = ddpm.forward_sample(x_0, t)
        >>> print(x_t.shape, noise.shape)  # torch.Size([4, 1, 28, 28])
    """

    def __init__(
        self,
        n_timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = 'linear',
        device: str = 'cuda'
    ) -> None:
        super().__init__()

        self.n_timesteps = n_timesteps
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 计算 Beta 调度序列
        if beta_schedule == 'linear':
            self.beta = self._linear_beta_schedule(
                beta_start, beta_end, n_timesteps
            )
        elif beta_schedule == 'cosine':
            self.beta = self._cosine_beta_schedule(n_timesteps)
        else:
            raise ValueError(
                f"不支持的 beta_schedule: {beta_schedule}。"
                f"请使用 'linear' 或 'cosine'。"
            )

        # 预计算所有需要的系数
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 前向过程系数: x_t = √α_bar * x_0 + √(1-α_bar) * ε
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # 反向过程系数
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)

        # 后验方差 β_tilde_t = (1 - α_bar_{t-1}) / (1 - α_bar_t) * β_t
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])
        self.beta_tilde = (
            (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar) * self.beta
        )
        # 数值稳定性：clamp 防止除零
        self.beta_tilde = torch.clamp(self.beta_tilde, min=1e-20)

        # 将所有参数移至指定设备
        self.beta = self.beta.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.alpha_bar = self.alpha_bar.to(self.device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(self.device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(
            self.device
        )
        self.sqrt_recip_alpha = self.sqrt_recip_alpha.to(self.device)
        self.beta_tilde = self.beta_tilde.to(self.device)

    def _linear_beta_schedule(
        self,
        beta_start: float,
        beta_end: float,
        n_timesteps: int
    ) -> torch.Tensor:
        """
        生成线性 Beta 调度序列。

        公式: β_t = β_start + (t / T) * (β_end - β_start)

        Args:
            beta_start (float): 起始 beta 值。
            beta_end (float): 终止 beta 值。
            n_timesteps (int): 时间步数。

        Returns:
            torch.Tensor: Beta 序列，shape (n_timesteps,)。
        """
        return torch.linspace(beta_start, beta_end, n_timesteps)

    def _cosine_beta_schedule(
        self,
        n_timesteps: int,
        s: float = 0.008
    ) -> torch.Tensor:
        """
        生成余弦 Beta 调度序列（Improved DDPM）。

        公式基于 alpha_bar 的余弦退火函数，生成质量更好的噪声调度。

        Args:
            n_timesteps (int): 时间步数。
            s (float, optional): 平滑参数。默认为 0.008。

        Returns:
            torch.Tensor: Beta 序列，shape (n_timesteps,)。
        """
        steps = n_timesteps + 1
        t = torch.linspace(0, n_timesteps, steps)
        alpha_bar = torch.cos(((t / n_timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return torch.clip(beta, 0.0001, 0.9999)

    def forward_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向加噪过程: x_0 -> x_t。

        根据重参数化技巧直接从 x_0 采样 x_t，无需逐步迭代：
        x_t = √α_bar_t * x_0 + √(1 - α_bar_t) * ε, 其中 ε ~ N(0, I)

        Args:
            x_0 (torch.Tensor): 原始无噪声数据。
                shape: (batch_size, channels, height, width)
            t (torch.Tensor): 时间步索引（0 到 T-1）。
                shape: (batch_size,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_t: 加噪后的数据，shape 与 x_0 相同。
                - noise: 采样的标准高斯噪声，shape 与 x_0 相同。

        Example:
            >>> ddpm = DDPM(n_timesteps=1000, beta_start=0.0001, beta_end=0.02)
            >>> x_0 = torch.randn(4, 1, 28, 28).cuda()
            >>> t = torch.randint(0, 1000, (4,)).cuda()
            >>> x_t, noise = ddpm.forward_sample(x_0, t)
        """
        # 采样标准高斯噪声
        noise = torch.randn_like(x_0)

        # 提取当前时间步的系数（shape: (batch_size, 1, 1, 1)）
        sqrt_alpha_bar_t = self._extract_coefficients(
            self.sqrt_alpha_bar, t, x_0.shape
        )
        sqrt_one_minus_alpha_bar_t = self._extract_coefficients(
            self.sqrt_one_minus_alpha_bar, t, x_0.shape
        )

        # 计算 x_t
        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def reverse_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        单步反向去噪: x_t -> x_{t-1}。

        根据 DDPM 反向过程公式，使用模型预测的噪声进行去噪：
        μ_θ(x_t, t) = 1/√α_t * (x_t - (β_t / √(1-α_bar_t)) * ε_θ(x_t, t))
        x_{t-1} = μ_θ(x_t, t) + σ_t * z, 其中 z ~ N(0, I) (t > 0)

        Args:
            x_t (torch.Tensor): 当前时间步的带噪数据。
                shape: (batch_size, channels, height, width)
            t (torch.Tensor): 当前时间步索引。
                shape: (batch_size,)
            condition (torch.Tensor): 条件张量（用于条件生成）。
                shape: (batch_size, channels, height, width) 或其他兼容形状
            model (nn.Module): 噪声预测模型（如 U-Net）。
                输入 (x_t, t, condition)，输出预测噪声。

        Returns:
            torch.Tensor: 去噪一步后的数据 x_{t-1}，shape 与 x_t 相同。

        Example:
            >>> model = SimpleUNet(in_channels=1, out_channels=1).cuda()
            >>> x_t = torch.randn(4, 1, 28, 28).cuda()
            >>> t = torch.tensor([100, 200, 300, 400]).cuda()
            >>> condition = torch.randn(4, 1, 28, 28).cuda()
            >>> x_t_minus_1 = ddpm.reverse_sample(x_t, t, condition, model)
        """
        # 使用模型预测噪声
        predicted_noise = model(x_t, t, condition)

        # 提取当前时间步的系数
        sqrt_recip_alpha_t = self._extract_coefficients(
            self.sqrt_recip_alpha, t, x_t.shape
        )
        beta_t = self._extract_coefficients(self.beta, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = self._extract_coefficients(
            self.sqrt_one_minus_alpha_bar, t, x_t.shape
        )
        beta_tilde_t = self._extract_coefficients(self.beta_tilde, t, x_t.shape)

        # 计算均值 μ_θ(x_t, t)
        mean = sqrt_recip_alpha_t * (
            x_t - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise
        )

        # 添加噪声（t > 0 时）
        noise = torch.randn_like(x_t)
        # 当 t == 0 时，不添加噪声
        noise_mask = (t > 0).float().view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(beta_tilde_t)

        x_t_minus_1 = mean + noise_mask * sigma_t * noise

        return x_t_minus_1

    def reverse_sample_loop(
        self,
        x_T: torch.Tensor,
        condition: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        完整反向采样循环: x_T -> x_0。

        从纯高斯噪声开始，逐步去噪 T 步，生成最终数据。

        Args:
            x_T (torch.Tensor): 初始纯高斯噪声（标准正态分布采样）。
                shape: (batch_size, channels, height, width)
            condition (torch.Tensor): 条件张量（用于条件生成）。
                shape 需与模型输入兼容
            model (nn.Module): 噪声预测模型。

        Returns:
            torch.Tensor: 生成的数据 x_0，shape 与 x_T 相同。

        Example:
            >>> model = SimpleUNet(in_channels=1, out_channels=1).cuda()
            >>> x_T = torch.randn(4, 1, 28, 28).cuda()
            >>> condition = torch.randn(4, 1, 28, 28).cuda()
            >>> x_0 = ddpm.reverse_sample_loop(x_T, condition, model)
        """
        x_t = x_T

        # 从 t = T-1 逐步去噪到 t = 0
        for time_step in reversed(range(self.n_timesteps)):
            # 创建时间步张量（batch内所有样本使用相同的时间步）
            t = torch.full(
                (x_t.shape[0],),
                time_step,
                dtype=torch.long,
                device=self.device
            )

            # 单步去噪
            x_t = self.reverse_sample(x_t, t, condition, model)

        return x_t

    def _extract_coefficients(
        self,
        coefficients: torch.Tensor,
        t: torch.Tensor,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """
        从系数序列中提取指定时间步的值，并广播到目标形状。

        Args:
            coefficients (torch.Tensor): 系数序列，shape (T,)。
            t (torch.Tensor): 时间步索引，shape (batch_size,)。
            target_shape (torch.Size): 目标张量形状，如 (batch_size, C, H, W)。

        Returns:
            torch.Tensor: 提取并广播后的系数，shape (batch_size, 1, 1, 1)。
        """
        batch_size = t.shape[0]
        extracted = coefficients.gather(-1, t)
        # 将 shape 从 (batch_size,) 变为 (batch_size, 1, 1, 1)
        return extracted.view(batch_size, *([1] * (len(target_shape) - 1)))


# ============================================================================
# 测试模块
# ============================================================================

if __name__ == "__main__":
    # 设置 UTF-8 编码以支持中文输出
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 导入 SimpleUNet 用于测试
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.unet import SimpleUNet

    print("=" * 70)
    print("DDPM 算法测试")
    print("=" * 70)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}\n")

    # 测试参数（MNIST 尺寸）
    batch_size = 4
    in_channels = 1
    out_channels = 1
    height = 28
    width = 28
    n_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02

    print("-" * 70)
    print("1. 初始化 DDPM (线性调度)")
    print("-" * 70)

    ddpm_linear = DDPM(
        n_timesteps=n_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule='linear',
        device=str(device)
    )
    print(f"✅ DDPM 初始化成功（线性调度）")
    print(f"   - 时间步数: {ddpm_linear.n_timesteps}")
    print(f"   - Beta 范围: [{ddpm_linear.beta[0].item():.6f}, "
          f"{ddpm_linear.beta[-1].item():.6f}]\n")

    print("-" * 70)
    print("2. 测试前向加噪过程 (forward_sample)")
    print("-" * 70)

    x_0 = torch.randn(batch_size, in_channels, height, width).to(device)
    t = torch.randint(0, n_timesteps, (batch_size,)).to(device)

    x_t, noise = ddpm_linear.forward_sample(x_0, t)

    print(f"输入 x_0 形状: {x_0.shape}")
    print(f"时间步 t: {t.cpu().numpy()}")
    print(f"输出 x_t 形状: {x_t.shape}")
    print(f"采样噪声形状: {noise.shape}")
    print(f"✅ 前向加噪测试通过！\n")

    print("-" * 70)
    print("3. 测试单步反向去噪 (reverse_sample)")
    print("-" * 70)

    # 创建 SimpleUNet 模型
    model = SimpleUNet(in_channels=in_channels, out_channels=out_channels).to(device)
    condition = torch.randn(batch_size, in_channels, height, width).to(device)

    with torch.no_grad():
        x_t_minus_1 = ddpm_linear.reverse_sample(x_t, t, condition, model)

    print(f"输入 x_t 形状: {x_t.shape}")
    print(f"时间步 t: {t.cpu().numpy()}")
    print(f"条件 condition 形状: {condition.shape}")
    print(f"输出 x_{{t-1}} 形状: {x_t_minus_1.shape}")
    print(f"✅ 单步反向去噪测试通过！\n")

    print("-" * 70)
    print("4. 测试完整反向采样循环 (reverse_sample_loop)")
    print("-" * 70)

    # 从纯噪声开始采样（使用较少的时间步以加快测试）
    ddpm_fast = DDPM(
        n_timesteps=50,  # 减少到 50 步以加快测试
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule='linear',
        device=str(device)
    )

    x_T = torch.randn(batch_size, in_channels, height, width).to(device)

    print(f"开始从 x_T 采样（共 {ddpm_fast.n_timesteps} 步）...")
    with torch.no_grad():
        x_0_generated = ddpm_fast.reverse_sample_loop(x_T, condition, model)

    print(f"输入 x_T 形状: {x_T.shape}")
    print(f"输出 x_0 形状: {x_0_generated.shape}")
    print(f"✅ 完整反向采样循环测试通过！\n")

    print("-" * 70)
    print("5. 测试余弦调度")
    print("-" * 70)

    ddpm_cosine = DDPM(
        n_timesteps=n_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule='cosine',
        device=str(device)
    )

    print(f"✅ DDPM 初始化成功（余弦调度）")
    print(f"   - Beta 范围: [{ddpm_cosine.beta[0].item():.6f}, "
          f"{ddpm_cosine.beta[-1].item():.6f}]")
    print(f"   - 前10个beta值: {ddpm_cosine.beta[:10].cpu().numpy()}\n")

    print("=" * 70)
    print("DDPM 算法所有测试通过！✅")
    print("=" * 70)
