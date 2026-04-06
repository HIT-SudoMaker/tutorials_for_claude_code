from typing import Optional, Tuple
import torch
import torch.nn as nn
from .ddpm import DDPM


class SDESolver(nn.Module):
    """
    支持随机采样的连续时间扩散采样器
    """

    def __init__(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = "linear",
    ) -> None:
        """
        构造SDE Solver并预计算噪声调度系数

        Args:
            timesteps:     总扩散步数T
            beta_start:    噪声调度起始值
            beta_end:      噪声调度终止值
            beta_schedule: 调度策略，支持"linear"和"cosine"
        """
        super().__init__()
        self.timesteps = timesteps

        # 借助DDPM预计算噪声调度系数
        _ddpm = DDPM(timesteps, beta_start, beta_end, beta_schedule)
        self.register_buffer("betas", _ddpm.betas)
        self.register_buffer("alphas_cumprod", _ddpm.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", _ddpm.sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", _ddpm.sqrt_one_minus_alphas_cumprod
        )

        # 反向SDE的系数
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), _ddpm.alphas_cumprod[:-1]]
        )
        # 后验均值系数，用于反向SDE中的漂移项
        self.register_buffer(
            "posterior_mean_coef1",
            _ddpm.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - _ddpm.alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(_ddpm.alphas) / (1.0 - _ddpm.alphas_cumprod),
        )
        # 后验方差，用于反向SDE中的扩散项
        posterior_variance = (
            _ddpm.betas * (1.0 - alphas_cumprod_prev) / (1.0 - _ddpm.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    def forward_sample(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
        """
        前向加噪：从x_0直接采样到x_t，逻辑与DDPM一致

        Args:
            x_0: 形状为 (B, C, H, W) 的原始数据
            t:   目标时间步索引

        Returns:
            形状与x_0相同的加噪数据x_t
        """
        noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].clamp(max=1.0)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].clamp(max=1.0)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def reverse_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: Optional[torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """
        反向SDE单步：使用Euler-Maruyama方法从x_t积分到x_{t-1}

        Args:
            x_t:       形状为 (B, C, H, W) 的当前带噪数据
            t:         当前时间步索引
            condition: 条件张量，为None时表示无条件生成
            model:     噪声预测模型

        Returns:
            形状与x_t相同的去噪结果x_{t-1}
        """
        batch_size = x_t.shape[0]
        t_tensor = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)
        predicted_noise = model(x_t, t_tensor, condition)

        # 使用与DDPM相同的后验分布公式
        # 反向SDE的离散化与DDPM的后验分布等价
        coef1 = self.posterior_mean_coef1[t].clamp(max=1.0)
        coef2 = self.posterior_mean_coef2[t].clamp(max=1.0)
        mean = coef1 * x_t - coef2 * predicted_noise

        if t == 0:
            return mean

        variance = self.posterior_variance[t]
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(variance.clamp(min=1e-20)) * noise

    def reverse_sample_loop(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """
        完整反向采样循环：从纯噪声开始使用反向SDE积分

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
    sde = SDESolver(
        timesteps=T,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )
    model = _DummyModel()
    shape = (2, 3, 32, 32)
    x_0 = torch.randn(shape)
    condition = torch.tensor([0, 1])

    # 测试forward_sample
    x_t = sde.forward_sample(x_0, t=500)
    assert x_t.shape == x_0.shape, f"forward_sample形状不匹配: {x_t.shape} vs {x_0.shape}"
    print(f"[SDESolver] forward_sample通过，x_t形状: {x_t.shape}")

    # 测试reverse_sample
    x_prev = sde.reverse_sample(x_t, t=500, condition=condition, model=model)
    assert x_prev.shape == x_t.shape, f"reverse_sample形状不匹配: {x_prev.shape} vs {x_t.shape}"
    print(f"[SDESolver] reverse_sample通过，x_prev形状: {x_prev.shape}")

    # 测试reverse_sample_loop
    sample = sde.reverse_sample_loop(shape, condition, model)
    assert sample.shape == shape, f"reverse_sample_loop形状不匹配: {sample.shape} vs {shape}"
    print(f"[SDESolver] reverse_sample_loop通过，sample形状: {sample.shape}")

    # 测试cosine调度
    sde_cos = SDESolver(
        timesteps=T,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="cosine",
    )
    x_t_cos = sde_cos.forward_sample(x_0, t=500)
    assert x_t_cos.shape == x_0.shape
    print("[SDESolver] cosine调度forward_sample通过")

    # 测试无条件生成
    sample_uncond = sde.reverse_sample_loop(shape, None, model)
    assert sample_uncond.shape == shape
    print("[SDESolver] 无条件reverse_sample_loop通过")

    print("[SDESolver] 全部接口形状验证通过")
