from typing import Optional, Tuple
import torch
import torch.nn as nn
from .ddpm import DDPM


class DDIM(nn.Module):
    """
    支持确定性与随机性采样的子序列扩散采样器
    """

    def __init__(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = "linear",
        eta: float = 0.0,
        steps: int = 50,
    ) -> None:
        """
        构造DDIM并预计算噪声调度与子序列时间步

        Args:
            timesteps:     训练总扩散步数T
            beta_start:    噪声调度起始值
            beta_end:      噪声调度终止值
            beta_schedule: 调度策略，支持"linear"和"cosine"
            eta:           随机性控制参数，0.0为完全确定性，1.0退化为DDPM
            steps:         采样步数，可小于训练timesteps
        """
        super().__init__()
        self.timesteps = timesteps
        self.eta = eta
        self.steps = steps

        # 借助DDPM预计算噪声调度系数
        _ddpm = DDPM(timesteps, beta_start, beta_end, beta_schedule)
        self.register_buffer("alphas_cumprod", _ddpm.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", _ddpm.sqrt_alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", _ddpm.sqrt_one_minus_alphas_cumprod
        )
        self.register_buffer("betas", _ddpm.betas)

        # 构建子序列时间步，均匀采样
        self.time_steps = torch.linspace(0, timesteps - 1, steps, dtype=torch.long)

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
        DDIM单步采样：从x_t预测x_{t-1}

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

        alpha_t = self.alphas_cumprod[t]
        sqrt_alpha_t = torch.sqrt(alpha_t.clamp(max=1.0))
        sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(max=1.0))

        # 预测x_0
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t.clamp(min=1e-10)
        x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        # 计算前一时间步的系数
        if t > 0:
            alpha_t_prev = self.alphas_cumprod[t - 1]
        else:
            alpha_t_prev = x_t.new_tensor(1.0)

        sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev.clamp(max=1.0))

        # 计算方向指向x_t的项
        sigma = (
            self.eta
            * torch.sqrt(
                (1.0 - alpha_t_prev) / (1.0 - alpha_t).clamp(min=1e-10)
                * (1.0 - alpha_t / alpha_t_prev.clamp(min=1e-10))
            )
        )
        dir_xt = torch.sqrt(1.0 - alpha_t_prev - sigma ** 2) * predicted_noise

        # 随机噪声项
        noise = torch.randn_like(x_t) if self.eta > 0 else torch.zeros_like(x_t)

        return sqrt_alpha_t_prev * x_0_pred + dir_xt + sigma * noise

    def reverse_sample_loop(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """
        完整反向采样循环：在子序列时间步上执行DDIM采样

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

        time_steps_list = self.time_steps.tolist()
        for i in reversed(range(len(time_steps_list))):
            t = int(time_steps_list[i])
            t_prev = int(time_steps_list[i - 1]) if i > 0 else 0

            batch_size = x.shape[0]
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor, condition)

            alpha_t = self.alphas_cumprod[t]
            sqrt_alpha_t = torch.sqrt(alpha_t.clamp(max=1.0))
            sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(max=1.0))

            x_0_pred = (x - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t.clamp(min=1e-10)
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)

            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else x.new_tensor(1.0)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev.clamp(max=1.0))

            sigma = (
                self.eta
                * torch.sqrt(
                    (1.0 - alpha_t_prev) / (1.0 - alpha_t).clamp(min=1e-10)
                    * (1.0 - alpha_t / alpha_t_prev.clamp(min=1e-10))
                )
            )
            dir_xt = torch.sqrt((1.0 - alpha_t_prev - sigma ** 2).clamp(min=0.0)) * predicted_noise
            noise = torch.randn_like(x) if self.eta > 0 else torch.zeros_like(x)

            x = sqrt_alpha_t_prev * x_0_pred + dir_xt + sigma * noise

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
    ddim = DDIM(
        timesteps=T,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        eta=0.0,
        steps=50,
    )
    model = _DummyModel()
    shape = (2, 3, 32, 32)
    x_0 = torch.randn(shape)
    condition = torch.tensor([0, 1])

    # 测试forward_sample
    x_t = ddim.forward_sample(x_0, t=500)
    assert x_t.shape == x_0.shape, f"forward_sample形状不匹配: {x_t.shape} vs {x_0.shape}"
    print(f"[DDIM] forward_sample通过，x_t形状: {x_t.shape}")

    # 测试reverse_sample
    x_prev = ddim.reverse_sample(x_t, t=500, condition=condition, model=model)
    assert x_prev.shape == x_t.shape, f"reverse_sample形状不匹配: {x_prev.shape} vs {x_t.shape}"
    print(f"[DDIM] reverse_sample通过，x_prev形状: {x_prev.shape}")

    # 测试reverse_sample_loop
    sample = ddim.reverse_sample_loop(shape, condition, model)
    assert sample.shape == shape, f"reverse_sample_loop形状不匹配: {sample.shape} vs {shape}"
    print(f"[DDIM] reverse_sample_loop通过，sample形状: {sample.shape}")

    # 测试eta=1.0
    ddim_stochastic = DDIM(
        timesteps=T,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        eta=1.0,
        steps=50,
    )
    sample_stoch = ddim_stochastic.reverse_sample_loop(shape, condition, model)
    assert sample_stoch.shape == shape
    print("[DDIM] eta=1.0随机性采样通过")

    # 测试cosine调度
    ddim_cos = DDIM(
        timesteps=T,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="cosine",
        eta=0.0,
        steps=20,
    )
    sample_cos = ddim_cos.reverse_sample_loop(shape, None, model)
    assert sample_cos.shape == shape
    print("[DDIM] cosine调度无条件采样通过")

    print("[DDIM] 全部接口形状验证通过")
