from pathlib import Path
from typing import Any, Dict, Tuple
import yaml
import torch
import torch.nn as nn
from models import build_model
from diffusion_implementations import build_diffusion


def load_config(config_path: str) -> Dict[str, Any]:
    """
    读取YAML配置文件并返回配置字典

    Args:
        config_path: YAML配置文件的路径，支持绝对路径或相对于项目根目录的路径

    Returns:
        解析后的配置字典，至少包含diffusion、model两个必要顶层键

    Raises:
        FileNotFoundError: 配置文件不存在时抛出
        ValueError: 配置文件内容为空或缺少必要顶层键时抛出
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"配置文件内容为空: {config_path}")

    required_top_keys = ["diffusion", "model"]
    missing_keys = [k for k in required_top_keys if k not in config]
    if missing_keys:
        raise ValueError(
            f"配置文件缺少必要顶层键: {missing_keys}"
        )

    return config


def build_from_config(
    config: Dict[str, Any],
) -> Tuple[nn.Module, nn.Module, Dict[str, Any]]:
    """
    根据配置字典构建扩散算法实例与噪声预测模型实例

    Args:
        config: 由load_config返回的完整配置字典，必须包含diffusion和model两个子字典

    Returns:
        包含三个元素的元组:
            - diffusion_instance: 构建好的扩散算法实例
            - model_instance: 构建好的噪声预测模型实例
            - config: 原始配置字典的引用，供调用方提取device、training等额外参数

    Raises:
        ValueError: diffusion或model子配置缺少必要字段时抛出
    """
    diffusion_config = config.get("diffusion", {})
    model_config = config.get("model", {})

    if not diffusion_config:
        raise ValueError("配置中缺少 diffusion 子配置")
    if not model_config:
        raise ValueError("配置中缺少 model 子配置")

    diffusion_instance = build_diffusion(diffusion_config)
    model_instance = build_model(model_config)

    return diffusion_instance, model_instance, config


def get_device(config: Dict[str, Any]) -> torch.device:
    """
    从配置字典中提取设备信息并返回torch.device对象

    Args:
        config: 由load_config返回的完整配置字典

    Returns:
        根据global.device配置构造的torch.device对象，
        若未指定device则默认使用cuda
    """
    global_config = config.get("global", {})
    device_str = global_config.get("device", "cuda")
    return torch.device(device_str)


if __name__ == "__main__":
    # 定位项目根目录下的config.yaml
    project_root = Path(__file__).resolve().parent
    config_path = str(project_root / "config.yaml")

    print(f"加载配置: {config_path}")
    config = load_config(config_path)
    print("配置加载成功")

    # 遍历所有 diffusion.implementation × model.type × condition.type 组合
    implementations = ["ddpm", "ddim", "sde_solver", "dpm_solver"]
    model_types = ["unet", "dit"]
    condition_types = ["none", "class"]

    passed = 0
    failed = 0

    for implementation in implementations:
        for model_type in model_types:
            for condition_type in condition_types:
                test_config = {
                    "diffusion": {
                        **config["diffusion"],
                        "implementation": implementation,
                    },
                    "model": {
                        **config["model"],
                        "type": model_type,
                        "condition": {
                            "type": condition_type,
                            "num_classes": 10,
                            "embedding_dim": 128,
                        },
                    },
                    "global": config.get("global", {}),
                }

                try:
                    diffusion_inst, model_inst, _ = build_from_config(
                        test_config
                    )
                    # 简单前向验证
                    x = torch.randn(1, 1, 32, 32)
                    t_tensor = torch.rand(1)
                    condition = (
                        torch.randint(0, 10, (1,))
                        if condition_type == "class"
                        else None
                    )
                    out = model_inst(x, t_tensor, condition)
                    assert out.shape == x.shape
                    print(
                        f"  [通过] impl={implementation}, model={model_type}, "
                        f"condition={condition_type}"
                    )
                    passed += 1
                except Exception as e:
                    print(
                        f"  [失败] impl={implementation}, model={model_type}, "
                        f"condition={condition_type} -> {e}"
                    )
                    failed += 1

    print(f"\n测试完成: {passed} 通过, {failed} 失败")

    # 测试快速失败行为
    print("\n--- 快速失败测试 ---")

    # 缺少diffusion配置
    try:
        build_from_config({"model": config["model"]})
        print("[失败] 未捕获缺少diffusion配置的错误")
    except ValueError as e:
        print(f"[通过] 缺少diffusion配置: {e}")

    # 缺少model配置
    try:
        build_from_config({"diffusion": config["diffusion"]})
        print("[失败] 未捕获缺少model配置的错误")
    except ValueError as e:
        print(f"[通过] 缺少model配置: {e}")

    # 不支持的implementation值（由build_diffusion抛出）
    try:
        bad_config = {
            "diffusion": {
                "implementation": "invalid_algo",
                "timesteps": 1000,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
            "model": config["model"],
        }
        build_from_config(bad_config)
        print("[失败] 未捕获不支持的implementation错误")
    except ValueError as e:
        print(f"[通过] 不支持的implementation: {e}")

    # 不支持的model type（由build_model抛出）
    try:
        bad_config = {
            "diffusion": config["diffusion"],
            "model": {
                "type": "invalid_model",
                "in_channels": 1,
                "out_channels": 1,
                "condition": {"type": "none"},
            },
        }
        build_from_config(bad_config)
        print("[失败] 未捕获不支持的model type错误")
    except ValueError as e:
        print(f"[通过] 不支持的model type: {e}")

    # 不存在的配置文件
    try:
        load_config("/nonexistent/path/config.yaml")
        print("[失败] 未捕获配置文件不存在的错误")
    except FileNotFoundError as e:
        print(f"[通过] 配置文件不存在: {e}")

    # get_device测试
    device = get_device(config)
    print(f"\nget_device返回: {device}")
