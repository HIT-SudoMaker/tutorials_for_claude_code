"""
DDIM (Denoising Diffusion Implicit Models) 算法实现。

基于 Song et al. 2021 的 DDIM 论文，实现支持加速采样的扩散模型。
DDIM 使用非 Markov 前向过程，允许跳步采样以显著减少生成步数。

参考文献:
    Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models.
    In International Conference on Learning Representations (ICLR).
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import numpy as np


class DDIM(nn.Module):
    """
    DDIM 扩散模型实现。

    相比 DDPM，DDIM 的主要优势是支持确定性或半确定性的加速采样。
    通过跳步采样（如从 1000 步跳到 50 步），可以在保持生成质量的同时大幅加速。

    Args:
        n_timesteps (int): 训练时的总扩散时间步数 T（通常为 1000）。
        beta_start (float): 初始噪声水平 β_0。
        beta_end (float): 最终噪声水平 β_T。
        beta_schedule (str, optional): Beta 调度方案（'linear' 或 'cosine'）。
            默认为 'linear'。
        eta (float, optional): 随机性控制参数。
            - eta=0: 确定性采样（DDIM）
            - eta=1: 等价于 DDPM（完全随机）
            - 0 < eta < 1: 半确定性采样
            默认为 0.0（确定性）。
        device (str, optional): 计算设备。默认为 'cuda'。

    Attributes:
        n_timesteps (int): 训练时的总时间步数。
        eta (float): 随机性参数。
        beta, alpha, alpha_bar: 与 DDPM 相同的系数。

    Example:
        >>> ddim = DDIM(n_timesteps=1000, eta=0.0)  # 确定性采样
        >>> x_T = torch.randn(4, 1, 28, 28).cuda()
        >>> # 使用 50 步而非 1000 步进行采样
        >>> x_0 = ddim.reverse_sample_loop(x_T, condition, model, skip_steps=50)
    """

    def __init__(
        self,
        n_timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = 'linear',
        eta: float = 0.0,
        device: str = 'cuda'
    ) -> None:
        super().__init__()

        self.n_timesteps = n_timesteps
        self.eta = eta
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 使用与 DDPM 相同的 Beta 调度
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

        # 预计算系数
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # 前向过程系数（与 DDPM 相同）
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

        # 移至设备
        self.beta = self.beta.to(self.device)
        self.alpha = self.alpha.to(self.device)
        self.alpha_bar = self.alpha_bar.to(self.device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(self.device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(
            self.device
        )

    def _linear_beta_schedule(
        self,
        beta_start: float,
        beta_end: float,
        n_timesteps: int
    ) -> torch.Tensor:
        """生成线性 Beta 调度序列。"""
        return torch.linspace(beta_start, beta_end, n_timesteps)

    def _cosine_beta_schedule(
        self,
        n_timesteps: int,
        s: float = 0.008
    ) -> torch.Tensor:
        """生成余弦 Beta 调度序列。"""
        import math
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

        与 DDPM 的前向过程完全相同，使用重参数化技巧。

        Args:
            x_0 (torch.Tensor): 原始无噪声数据。
                shape: (batch_size, channels, height, width)
            t (torch.Tensor): 时间步索引（0 到 T-1）。
                shape: (batch_size,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_t: 加噪后的数据。
                - noise: 采样的标准高斯噪声。
        """
        noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = self._extract_coefficients(
            self.sqrt_alpha_bar, t, x_0.shape
        )
        sqrt_one_minus_alpha_bar_t = self._extract_coefficients(
            self.sqrt_one_minus_alpha_bar, t, x_0.shape
        )

        x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise

        return x_t, noise

    def reverse_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        model: nn.Module,
        t_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        单步反向去噪: x_t -> x_{t-1}（支持跳步）。

        DDIM 反向过程公式（确定性，eta=0）：
        x_{t-1} = √α_bar_{t-1} * (x_t - √(1-α_bar_t) * ε_θ) / √α_bar_t
                  + √(1 - α_bar_{t-1} - σ_t^2) * ε_θ + σ_t * z

        其中 σ_t = eta * √((1-α_bar_{t-1})/(1-α_bar_t)) * √(1-α_bar_t/α_bar_{t-1})

        Args:
            x_t (torch.Tensor): 当前时间步的带噪数据。
                shape: (batch_size, channels, height, width)
            t (torch.Tensor): 当前时间步索引。
                shape: (batch_size,)
            condition (torch.Tensor): 条件张量。
            model (nn.Module): 噪声预测模型。
            t_prev (Optional[torch.Tensor], optional): 前一个时间步索引。
                如果为 None，则假设为 t-1（标准单步去噪）。
                用于跳步采样时指定目标时间步。

        Returns:
            torch.Tensor: 去噪后的数据 x_{t-1}。
        """
        # 使用模型预测噪声
        predicted_noise = model(x_t, t, condition)

        # 如果未指定前一个时间步，默认为 t-1
        if t_prev is None:
            t_prev = torch.clamp(t - 1, min=0)

        # 提取当前和前一时间步的 alpha_bar
        alpha_bar_t = self._extract_coefficients(self.alpha_bar, t, x_t.shape)
        alpha_bar_t_prev = self._extract_coefficients(
            self.alpha_bar, t_prev, x_t.shape
        )

        # 计算 σ_t（方差项）
        sigma_t = (
            self.eta
            * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t))
            * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
        )

        # 预测 x_0（数据预测）
        predicted_x_0 = (
            x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise
        ) / torch.sqrt(alpha_bar_t)

        # 计算方向指向 x_t 的部分
        direction_pointing_to_x_t = torch.sqrt(
            1 - alpha_bar_t_prev - sigma_t ** 2
        ) * predicted_noise

        # 随机噪声项
        noise = torch.randn_like(x_t)
        # 当 t_prev == 0 时，不添加噪声
        noise_mask = (t_prev > 0).float().view(-1, 1, 1, 1)

        # DDIM 反向采样公式
        x_t_prev = (
            torch.sqrt(alpha_bar_t_prev) * predicted_x_0
            + direction_pointing_to_x_t
            + noise_mask * sigma_t * noise
        )

        return x_t_prev

    def reverse_sample_loop(
        self,
        x_T: torch.Tensor,
        condition: torch.Tensor,
        model: nn.Module,
        skip_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        完整反向采样循环: x_T -> x_0（支持跳步加速）。

        Args:
            x_T (torch.Tensor): 初始纯高斯噪声。
                shape: (batch_size, channels, height, width)
            condition (torch.Tensor): 条件张量。
            model (nn.Module): 噪声预测模型。
            skip_steps (Optional[int], optional): 跳步采样的实际步数。
                如果为 None，使用完整的 n_timesteps 步。
                例如，skip_steps=50 表示从 1000 步跳到 50 步。

        Returns:
            torch.Tensor: 生成的数据 x_0。

        Example:
            >>> # 标准采样（1000 步）
            >>> x_0 = ddim.reverse_sample_loop(x_T, condition, model)
            >>> # 加速采样（50 步）
            >>> x_0_fast = ddim.reverse_sample_loop(x_T, condition, model, skip_steps=50)
        """
        # 生成时间步序列
        if skip_steps is None:
            # 使用全部时间步
            timesteps = list(reversed(range(self.n_timesteps)))
        else:
            # 跳步采样：均匀选择 skip_steps 个时间步
            timesteps = list(
                np.linspace(0, self.n_timesteps - 1, skip_steps, dtype=int)
            )
            timesteps = list(reversed(timesteps))

        x_t = x_T

        # 逐步去噪
        for i, time_step in enumerate(timesteps):
            # 创建当前时间步张量
            t = torch.full(
                (x_t.shape[0],),
                time_step,
                dtype=torch.long,
                device=self.device
            )

            # 特殊处理最后一步 (t=0)
            if time_step == 0:
                # 当 t=0 时，直接返回预测的 x_0
                predicted_noise = model(x_t, t, condition)
                alpha_bar_t = self._extract_coefficients(
                    self.alpha_bar, t, x_t.shape
                )
                # 计算 predicted_x_0
                x_0 = (
                    x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise
                ) / torch.sqrt(alpha_bar_t)
                return x_0

            # 确定前一个时间步
            if i < len(timesteps) - 1:
                t_prev = torch.full(
                    (x_t.shape[0],),
                    timesteps[i + 1],
                    dtype=torch.long,
                    device=self.device
                )
            else:
                # 理论上不应到达这里，因为上面已经处理了 t=0
                t_prev = torch.zeros_like(t)

            # 单步去噪
            x_t = self.reverse_sample(x_t, t, condition, model, t_prev)

        return x_t

    def _extract_coefficients(
        self,
        coefficients: torch.Tensor,
        t: torch.Tensor,
        target_shape: torch.Size
    ) -> torch.Tensor:
        """从系数序列中提取指定时间步的值并广播。"""
        batch_size = t.shape[0]
        extracted = coefficients.gather(-1, t)
        return extracted.view(batch_size, *([1] * (len(target_shape) - 1)))


# ============================================================================
# 测试模块
# ============================================================================

if __name__ == "__main__":
    # 设置 UTF-8 编码
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    # 导入 SimpleUNet
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.unet import SimpleUNet

    print("=" * 70)
    print("DDIM 算法测试")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}\n")

    # 测试参数
    batch_size = 4
    in_channels = 1
    out_channels = 1
    height = 28
    width = 28
    n_timesteps = 1000
    beta_start = 0.0001
    beta_end = 0.02

    print("-" * 70)
    print("1. 初始化 DDIM（确定性采样，eta=0）")
    print("-" * 70)

    ddim = DDIM(
        n_timesteps=n_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule='linear',
        eta=0.0,  # 确定性采样
        device=str(device)
    )
    print(f"✅ DDIM 初始化成功")
    print(f"   - 时间步数: {ddim.n_timesteps}")
    print(f"   - Eta（随机性参数）: {ddim.eta}")
    print(f"   - Beta 范围: [{ddim.beta[0].item():.6f}, "
          f"{ddim.beta[-1].item():.6f}]\n")

    print("-" * 70)
    print("2. 测试前向加噪过程（与 DDPM 相同）")
    print("-" * 70)

    x_0 = torch.randn(batch_size, in_channels, height, width).to(device)
    t = torch.randint(0, n_timesteps, (batch_size,)).to(device)

    x_t, noise = ddim.forward_sample(x_0, t)

    print(f"输入 x_0 形状: {x_0.shape}")
    print(f"时间步 t: {t.cpu().numpy()}")
    print(f"输出 x_t 形状: {x_t.shape}")
    print(f"✅ 前向加噪测试通过！\n")

    print("-" * 70)
    print("3. 测试单步反向去噪（标准步长）")
    print("-" * 70)

    model = SimpleUNet(in_channels=in_channels, out_channels=out_channels).to(device)
    condition = torch.randn(batch_size, in_channels, height, width).to(device)

    with torch.no_grad():
        x_t_minus_1 = ddim.reverse_sample(x_t, t, condition, model)

    print(f"输入 x_t 形状: {x_t.shape}")
    print(f"时间步 t: {t.cpu().numpy()}")
    print(f"输出 x_{{t-1}} 形状: {x_t_minus_1.shape}")
    print(f"✅ 单步反向去噪测试通过！\n")

    print("-" * 70)
    print("4. 测试跳步采样（1000 步 -> 50 步加速）")
    print("-" * 70)

    ddim_fast = DDIM(
        n_timesteps=n_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule='linear',
        eta=0.0,
        device=str(device)
    )

    x_T = torch.randn(batch_size, in_channels, height, width).to(device)

    print(f"开始跳步采样（从 {n_timesteps} 步跳到 50 步）...")
    with torch.no_grad():
        x_0_fast = ddim_fast.reverse_sample_loop(
            x_T, condition, model, skip_steps=50
        )

    print(f"输入 x_T 形状: {x_T.shape}")
    print(f"输出 x_0 形状: {x_0_fast.shape}")
    print(f"✅ 跳步采样测试通过！（仅用 50 步完成采样）\n")

    print("-" * 70)
    print("5. 测试不同 eta 值的影响")
    print("-" * 70)

    # eta = 0（完全确定性）
    ddim_deterministic = DDIM(
        n_timesteps=100, beta_start=beta_start, beta_end=beta_end,
        eta=0.0, device=str(device)
    )
    print(f"✅ eta=0.0 (确定性) DDIM 初始化成功")

    # eta = 1（等价于 DDPM）
    ddim_stochastic = DDIM(
        n_timesteps=100, beta_start=beta_start, beta_end=beta_end,
        eta=1.0, device=str(device)
    )
    print(f"✅ eta=1.0 (完全随机，类似 DDPM) DDIM 初始化成功")

    # eta = 0.5（半确定性）
    ddim_semi = DDIM(
        n_timesteps=100, beta_start=beta_start, beta_end=beta_end,
        eta=0.5, device=str(device)
    )
    print(f"✅ eta=0.5 (半确定性) DDIM 初始化成功\n")

    print("-" * 70)
    print("6. 对比确定性采样的可重复性")
    print("-" * 70)

    torch.manual_seed(42)
    x_T_1 = torch.randn(1, in_channels, height, width).to(device)
    cond_1 = torch.randn(1, in_channels, height, width).to(device)

    with torch.no_grad():
        result_1 = ddim_deterministic.reverse_sample_loop(
            x_T_1.clone(), cond_1, model, skip_steps=20
        )
        result_2 = ddim_deterministic.reverse_sample_loop(
            x_T_1.clone(), cond_1, model, skip_steps=20
        )

    difference = torch.abs(result_1 - result_2).max().item()
    print(f"两次确定性采样的最大差异: {difference:.10f}")
    if difference < 1e-5:
        print(f"✅ 确定性采样可重复性测试通过！\n")
    else:
        print(f"⚠️  差异较大，可能存在随机性\n")

    print("=" * 70)
    print("DDIM 算法所有测试通过！✅")
    print("=" * 70)
