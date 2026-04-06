from typing import Any, Dict
import torch.nn as nn
from .ddim import DDIM
from .ddpm import DDPM
from .sde_solver import SDESolver
from .dpm_solver import DPMSolver


def build_diffusion(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置字典构建扩散算法实例

    Args:
        config: diffusion配置字典，必须包含implementation字段，
            其他字段根据具体算法传递

    Returns:
        构建好的扩散算法实例

    Raises:
        ValueError: 不支持的implementation值或缺少必要配置
    """
    implementation = config.get("implementation")
    if implementation is None:
        raise ValueError("配置中缺少diffusion.implementation字段")

    # 所有算法共享的基础参数
    required_keys = ["timesteps", "beta_start", "beta_end"]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"diffusion配置缺少必要字段: {missing_keys}")

    # 校验数值范围
    timesteps = config["timesteps"]
    if not isinstance(timesteps, int) or timesteps <= 0:
        raise ValueError(
            f"diffusion.timesteps必须为正整数，当前值: {timesteps}"
        )

    for field in ["beta_start", "beta_end"]:
        value = config[field]
        if not isinstance(value, (int, float)) or not (0 < value < 1):
            raise ValueError(
                f"diffusion.{field}必须为0到1之间的数值，当前值: {value}"
            )

    if config["beta_start"] >= config["beta_end"]:
        raise ValueError(
            f"diffusion.beta_start必须小于beta_end，"
            f"当前值: {config['beta_start']} >= {config['beta_end']}"
        )

    common_kwargs = {
        "timesteps": config["timesteps"],
        "beta_start": config["beta_start"],
        "beta_end": config["beta_end"],
        "beta_schedule": config.get("beta_schedule", "linear"),
    }

    if implementation == "ddpm":
        return DDPM(**common_kwargs)
    elif implementation == "ddim":
        ddim_config = config.get("ddim", {})
        return DDIM(
            **common_kwargs,
            eta=ddim_config.get("eta", 0.0),
            steps=ddim_config.get("steps", 50),
        )
    elif implementation == "sde_solver":
        return SDESolver(**common_kwargs)
    elif implementation == "dpm_solver":
        dpm_config = config.get("dpm_solver", {})
        order = dpm_config.get("order", 2)
        if order not in {1, 2}:
            raise ValueError(
                f"不支持的dpm_solver.order: {order}，支持的值为: [1, 2]"
            )
        return DPMSolver(
            **common_kwargs,
            order=order,
            steps=dpm_config.get("steps", 20),
        )
    else:
        supported = ["ddpm", "ddim", "sde_solver", "dpm_solver"]
        raise ValueError(
            f"不支持的diffusion.implementation: {implementation}，"
            f"支持的值为: {supported}"
        )
