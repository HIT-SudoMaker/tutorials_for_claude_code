from typing import Any, Dict
import torch.nn as nn
from .dit import DiT
from .unet import UNet


_MODEL_REGISTRY: Dict[str, type] = {
    "unet": UNet,
    "dit": DiT,
}


def build_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    根据配置字典构建噪声预测模型

    Args:
        model_config: 模型配置字典，必须包含以下键:
            - type: 模型类型，支持 "unet"、"dit"
            - in_channels: 输入通道数，必须与out_channels一致
            - out_channels: 输出通道数，必须与in_channels一致
            - condition: 条件配置字典，包含 type、num_classes、embedding_dim

    Returns:
        构建好的模型实例

    Raises:
        ValueError: model.type为不支持的类型、condition.type非法、
            缺少必要字段、或in_channels与out_channels不一致时抛出
    """
    # 校验必要字段
    for key in ["type", "in_channels", "out_channels"]:
        if key not in model_config:
            raise ValueError(f"model配置缺少必要字段: '{key}'")

    model_type = model_config.get("type")
    if model_type not in _MODEL_REGISTRY:
        supported = sorted(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"不支持的模型类型: '{model_type}'，"
            f"支持的类型: {supported}"
        )

    # 强制 out_channels == in_channels，保证输出形状与 x_t 一致
    if model_config["in_channels"] != model_config["out_channels"]:
        raise ValueError(
            f"in_channels({model_config['in_channels']}) "
            f"必须与out_channels({model_config['out_channels']})一致，"
            f"模型输出形状必须与x_t一致"
        )

    condition_config = model_config.get("condition", {})
    condition_type = condition_config.get("type", "none")

    if condition_type not in {"none", "class"}:
        raise ValueError(
            f"不支持的condition.type: '{condition_type}'，"
            f"支持的类型: ['none', 'class']"
        )

    num_classes = condition_config.get("num_classes", 10)
    embedding_dim = condition_config.get("embedding_dim", 128)

    model_cls = _MODEL_REGISTRY[model_type]
    return model_cls(
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        condition_type=condition_type,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
    )


__all__ = ["UNet", "DiT", "build_model"]


if __name__ == "__main__":
    import torch

    test_config = {
        "type": "unet",
        "in_channels": 1,
        "out_channels": 1,
        "condition": {
            "type": "class",
            "num_classes": 10,
            "embedding_dim": 128,
        },
    }

    model = build_model(test_config)
    x = torch.randn(2, 1, 32, 32)
    t = torch.randn(2)
    cond = torch.randint(0, 10, (2,))
    out = model(x, t, cond)
    assert out.shape == x.shape
    print("build_model class模式测试通过")

    test_config["condition"]["type"] = "none"
    model = build_model(test_config)
    out = model(x, t, None)
    assert out.shape == x.shape
    print("build_model none模式测试通过")

    try:
        build_model({"type": "invalid_type", "in_channels": 1, "out_channels": 1, "condition": {"type": "none"}})
        raise AssertionError("未捕获不支持的模型类型")
    except ValueError:
        print("build_model 快速失败测试通过")
