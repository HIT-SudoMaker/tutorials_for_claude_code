from typing import Optional, Tuple
import torch
import torch.nn as nn
from .ddpm import DDPM


class DPMSolver(nn.Module):
    """
    支持少步高质量采样的加速扩散采样器
    """

    def __init__(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = "linear",
        order: int = 2,
        steps: int = 20,
    ) -> None:
        """
        构造DPM-Solver并预计算噪声调度与子序列时间步

        Args:
            timesteps:     训练总扩散步数T
            beta_start:    噪声调度起始值
            beta_end:      噪声调度终止值
            beta_schedule: 调度策略，支持"linear"和"cosine"
            order:         多步求解器阶数，支持1和2
            steps:         采样步数
        """
        super().__init__()
        self.timesteps = timesteps
        self.order = order
        self.steps = steps

        # 借助DDPM预计算噪声调度系数
        _ddpm = DDPM(timesteps, beta_start, beta_end, beta_schedule)
        self.register_buffer("alphas_cumprod", _ddpm.alphas_cumprod)
        self.register_buffer("betas", _ddpm.betas)

        # 构建子序列时间步
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
        alpha_t = self.alphas_cumprod[t].clamp(max=1.0)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(max=1.0))
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

    def _predict_x0(
        self,
        x_t: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        从预测噪声还原x_0

        Args:
            x_t:                当前带噪数据
            predicted_noise:    模型预测的噪声
            t:                  当前时间步索引

        Returns:
            预测的x_0
        """
        alpha_t = self.alphas_cumprod[t].clamp(max=1.0)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(max=1.0))
        return (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t.clamp(min=1e-10)

    def _dpm_solver_step(
        self,
        x: torch.Tensor,
        predicted_noise: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> torch.Tensor:
        """
        DPM-Solver一阶/二阶单步去噪

        Args:
            x:                  当前状态x_t
            predicted_noise:    模型预测的噪声
            t:                  当前时间步索引
            t_prev:             目标时间步索引

        Returns:
            去噪后的x_{t_prev}
        """
        alpha_t = self.alphas_cumprod[t].clamp(max=1.0)
        alpha_t_prev = self.alphas_cumprod[t_prev].clamp(max=1.0)

        # 预测x_0
        x_0 = self._predict_x0(x, predicted_noise, t)
        x_0 = x_0.clamp(-1.0, 1.0)

        if self.order == 1:
            # 一阶DPM-Solver
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t_prev = torch.sqrt((1.0 - alpha_t_prev).clamp(max=1.0))
            return sqrt_alpha_t_prev * x_0 + sqrt_one_minus_alpha_t_prev * predicted_noise
        else:
            # 二阶DPM-Solver：使用log-SNR空间的多步校正公式
            d_t = (torch.sqrt(alpha_t_prev) * x_0 - torch.sqrt(alpha_t) * x) / (
                alpha_t_prev - alpha_t
            ).clamp(min=1e-10)

            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            x_prev = (
                sqrt_alpha_t_prev * x_0
                + (sqrt_alpha_t_prev - torch.sqrt(alpha_t)) * d_t
            )
            return x_prev

    def reverse_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        condition: Optional[torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """
        DPM-Solver单步去噪：使用多步公式从x_t计算x_{t-1}

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

        # 确定前一个时间步
        t_prev = max(0, t - 1)

        return self._dpm_solver_step(x_t, predicted_noise, t, t_prev)

    def reverse_sample_loop(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """
        完整反向采样循环：在子序列时间步上执行DPM-Solver采样

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
            # 确定前一个子序列时间步
            if i > 0:
                t_prev = int(time_steps_list[i - 1])
            else:
                t_prev = 0

            batch_size = x.shape[0]
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor, condition)

            x = self._dpm_solver_step(x, predicted_noise, t, t_prev)

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
    dpm = DPMSolver(
        timesteps=T,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        order=2,
        steps=20,
    )
    model = _DummyModel()
    shape = (2, 3, 32, 32)
    x_0 = torch.randn(shape)
    condition = torch.tensor([0, 1])

    # 测试forward_sample
    x_t = dpm.forward_sample(x_0, t=500)
    assert x_t.shape == x_0.shape, f"forward_sample形状不匹配: {x_t.shape} vs {x_0.shape}"
    print(f"[DPMSolver] forward_sample通过，x_t形状: {x_t.shape}")

    # 测试reverse_sample
    x_prev = dpm.reverse_sample(x_t, t=500, condition=condition, model=model)
    assert x_prev.shape == x_t.shape, f"reverse_sample形状不匹配: {x_prev.shape} vs {x_t.shape}"
    print(f"[DPMSolver] reverse_sample通过，x_prev形状: {x_prev.shape}")

    # 测试reverse_sample_loop
    sample = dpm.reverse_sample_loop(shape, condition, model)
    assert sample.shape == shape, f"reverse_sample_loop形状不匹配: {sample.shape} vs {shape}"
    print(f"[DPMSolver] reverse_sample_loop通过，sample形状: {sample.shape}")

    # 测试一阶
    dpm_order1 = DPMSolver(
        timesteps=T,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        order=1,
        steps=20,
    )
    sample_o1 = dpm_order1.reverse_sample_loop(shape, condition, model)
    assert sample_o1.shape == shape
    print("[DPMSolver] order=1采样通过")

    # 测试cosine调度
    dpm_cos = DPMSolver(
        timesteps=T,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="cosine",
        order=2,
        steps=20,
    )
    sample_cos = dpm_cos.reverse_sample_loop(shape, None, model)
    assert sample_cos.shape == shape
    print("[DPMSolver] cosine调度无条件采样通过")

    print("[DPMSolver] 全部接口形状验证通过")
