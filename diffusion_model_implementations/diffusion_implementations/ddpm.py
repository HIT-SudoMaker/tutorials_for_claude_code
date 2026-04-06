import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class DDPM(nn.Module):
    """
    提供前向加噪与反向去噪的扩散采样接口
    """

    def __init__(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = "linear",
    ) -> None:
        """
        构造DDPM并预计算噪声调度系数

        Args:
            timesteps:     总扩散步数T
            beta_start:    噪声调度起始值
            beta_end:      噪声调度终止值
            beta_schedule: 调度策略，支持"linear"和"cosine"
        """
        super().__init__()
        self.timesteps = timesteps

        betas = self._build_schedule(timesteps, beta_start, beta_end, beta_schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 预计算前向过程所需系数
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # 预计算反向过程所需系数
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]]
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _build_schedule(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        schedule: str,
    ) -> torch.Tensor:
        """
        构建噪声调度序列

        Args:
            timesteps:  总步数
            beta_start: 起始值
            beta_end:   终止值
            schedule:   调度策略

        Returns:
            长度为timesteps的beta张量

        Raises:
            ValueError: 不支持的调度策略
        """
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, timesteps)
        elif schedule == "cosine":
            steps = timesteps + 1
            offset = 0.008
            step_indices = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((step_indices / timesteps) + offset) / (1 + offset) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, beta_start, beta_end)
        else:
            raise ValueError(
                f"不支持的beta_schedule: {schedule}，仅支持 linear 和 cosine"
            )

    def _extract(self, coeffs: torch.Tensor, t: int) -> torch.Tensor:
        """
        安全提取时间步系数并 clamp 上界，防止浮点累积误差导致系数超过1.0

        Args:
            coeffs: 形状为 (T,) 的系数张量
            t:      时间步索引

        Returns:
            标量张量
        """
        return coeffs[t].clamp(max=1.0)

    def forward_sample(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
        """
        前向加噪：从x_0直接采样到x_t

        Args:
            x_0: 形状为 (B, C, H, W) 的原始数据
            t:   目标时间步索引

        Returns:
            形状与x_0相同的加噪数据x_t
        """
        noise = torch.randn_like(x_0)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def reverse_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: Optional[torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """
        单步反向去噪：给定x_t，预测噪声并计算x_{t-1}的分布后采样

        Args:
            x_t:       形状为 (B, C, H, W) 的当前带噪数据
            t:         当前时间步索引
            condition: 条件张量，为None时表示无条件生成
            model:     噪声预测模型

        Returns:
            形状与x_t相同的去噪结果x_{t-1}
        """
        batch_size = x_t.shape[0]
        # 模型forward要求t为形状(B,)的张量
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        predicted_noise = model(x_t, t_tensor, condition)

        coef1 = self._extract(self.posterior_mean_coef1, t)
        coef2 = self._extract(self.posterior_mean_coef2, t)
        mean = coef1 * x_t - coef2 * predicted_noise

        if t == 0:
            return mean

        variance = self._extract(self.posterior_variance, t)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(variance) * noise

    def reverse_sample_loop(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """
        完整反向采样循环：从纯噪声逐步去噪到x_0

        Args:
            shape:     目标张量形状，如 (B, C, H, W)
            condition: 条件张量，为None时表示无条件生成
            model:     噪声预测模型

        Returns:
            形状为shape的生成样本
        """
        # 从模型参数推断设备，模型无参数时默认使用cpu
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.timesteps)):
            x = self.reverse_sample(x, t, condition, model)

        return x


if __name__ == "__main__":
    # 验证接口形状的轻量测试
    class _DummyModel(nn.Module):
        def forward(
            self,
            x_t: torch.Tensor,
            t: torch.Tensor,
            condition: Optional[torch.Tensor],
        ) -> torch.Tensor:
            return torch.randn_like(x_t)

    T = 1000
    ddpm = DDPM(timesteps=T, beta_start=0.0001, beta_end=0.02, beta_schedule="linear")
    model = _DummyModel()
    shape = (2, 3, 32, 32)
    x_0 = torch.randn(shape)
    condition = torch.tensor([0, 1])

    # 测试forward_sample
    x_t = ddpm.forward_sample(x_0, t=500)
    assert x_t.shape == x_0.shape, f"forward_sample形状不匹配: {x_t.shape} vs {x_0.shape}"
    print(f"[DDPM] forward_sample通过，x_t形状: {x_t.shape}")

    # 测试reverse_sample
    x_prev = ddpm.reverse_sample(x_t, t=500, condition=condition, model=model)
    assert x_prev.shape == x_t.shape, f"reverse_sample形状不匹配: {x_prev.shape} vs {x_t.shape}"
    print(f"[DDPM] reverse_sample通过，x_prev形状: {x_prev.shape}")

    # 测试reverse_sample_loop
    sample = ddpm.reverse_sample_loop(shape, condition, model)
    assert sample.shape == shape, f"reverse_sample_loop形状不匹配: {sample.shape} vs {shape}"
    print(f"[DDPM] reverse_sample_loop通过，sample形状: {sample.shape}")

    # 测试cosine调度
    ddpm_cos = DDPM(timesteps=T, beta_start=0.0001, beta_end=0.02, beta_schedule="cosine")
    x_t_cos = ddpm_cos.forward_sample(x_0, t=500)
    assert x_t_cos.shape == x_0.shape
    print("[DDPM] cosine调度forward_sample通过")

    # 测试无条件生成
    sample_uncond = ddpm.reverse_sample_loop(shape, None, model)
    assert sample_uncond.shape == shape
    print("[DDPM] 无条件reverse_sample_loop通过")

    print("[DDPM] 全部接口形状验证通过")
