import sys
import unittest
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml
import torch

# 将项目根目录添加到搜索路径，确保可以导入runtime等模块
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models import build_model
from diffusion_implementations import build_diffusion
from runtime import build_from_config, get_device, load_config

# config.yaml的绝对路径
_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config.yaml")


def _make_full_config(
    implementation: str = "ddpm",
    model_type: str = "unet",
    condition_type: str = "none",
    in_channels: int = 1,
    out_channels: int = 1,
    num_classes: int = 10,
    embedding_dim: int = 128,
) -> Dict[str, Any]:
    """
    构造一份完整的测试用配置字典

    Args:
        implementation:   diffusion算法名称
        model_type:       模型类型名称
        condition_type:   条件模式，"none"或"class"
        in_channels:      输入通道数
        out_channels:     输出通道数
        num_classes:      分类条件类别数
        embedding_dim:    条件嵌入维度

    Returns:
        可直接传给build_from_config的配置字典
    """
    return {
        "diffusion": {
            "implementation": implementation,
            "timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "ddim": {"eta": 0.0, "steps": 50},
            "dpm_solver": {"order": 2, "steps": 20},
        },
        "model": {
            "type": model_type,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "condition": {
                "type": condition_type,
                "num_classes": num_classes,
                "embedding_dim": embedding_dim,
            },
        },
        "global": {"device": "cpu"},
    }


class TestLoadConfig(unittest.TestCase):
    """
    配置加载相关测试
    """

    def test_normal_load(self) -> None:
        """
        正常加载config.yaml，返回包含diffusion和model键的字典
        """
        config = load_config(_CONFIG_PATH)
        self.assertIsInstance(config, dict)
        self.assertIn("diffusion", config)
        self.assertIn("model", config)

    def test_file_not_found(self) -> None:
        """
        文件不存在时抛出FileNotFoundError
        """
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_missing_required_top_keys(self) -> None:
        """
        缺少必要顶层键时抛出ValueError
        """
        bad_data = {"model": {"type": "test"}}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            yaml.dump(bad_data, f)
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_config(tmp_path)
            self.assertIn("diffusion", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_empty_config_raises(self) -> None:
        """
        配置文件内容为空时抛出ValueError
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                load_config(tmp_path)
            self.assertIn("内容为空", str(ctx.exception))
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestBuildModel(unittest.TestCase):
    """
    模型构建相关测试
    """

    def test_build_unet(self) -> None:
        """
        构建unet模型成功返回nn.Module实例
        """
        cfg = _make_full_config(model_type="unet")
        model = build_model(cfg["model"])
        self.assertIsInstance(model, torch.nn.Module)

    def test_build_dit(self) -> None:
        """
        构建dit模型成功返回nn.Module实例
        """
        cfg = _make_full_config(model_type="dit")
        model = build_model(cfg["model"])
        self.assertIsInstance(model, torch.nn.Module)

    def test_unsupported_model_type(self) -> None:
        """
        不支持的模型类型快速失败并抛出ValueError
        """
        bad_config: Dict[str, Any] = {
            "type": "invalid_model",
            "in_channels": 1,
            "out_channels": 1,
            "condition": {"type": "none"},
        }
        with self.assertRaises(ValueError) as ctx:
            build_model(bad_config)
        self.assertIn("不支持的模型类型", str(ctx.exception))

    def test_in_channels_passed_correctly(self) -> None:
        """
        in_channels参数正确传递，模型接受对应通道数的输入
        """
        cfg = _make_full_config(in_channels=3, out_channels=3)
        model = build_model(cfg["model"])
        x = torch.randn(1, 3, 32, 32)
        t = torch.rand(1)
        out = model(x, t, None)
        self.assertEqual(out.shape, x.shape)

    def test_out_channels_passed_correctly(self) -> None:
        """
        out_channels参数正确传递，输出通道数与配置一致
        """
        cfg = _make_full_config(in_channels=3, out_channels=3)
        model = build_model(cfg["model"])
        x = torch.randn(1, 3, 32, 32)
        t = torch.rand(1)
        out = model(x, t, None)
        self.assertEqual(out.shape[1], 3)


class TestBuildDiffusion(unittest.TestCase):
    """
    扩散算法构建相关测试
    """

    def test_build_ddpm(self) -> None:
        """
        构建DDPM算法实例成功
        """
        cfg = _make_full_config(implementation="ddpm")
        algo = build_diffusion(cfg["diffusion"])
        self.assertIsInstance(algo, torch.nn.Module)

    def test_build_ddim(self) -> None:
        """
        构建DDIM算法实例成功
        """
        cfg = _make_full_config(implementation="ddim")
        algo = build_diffusion(cfg["diffusion"])
        self.assertIsInstance(algo, torch.nn.Module)

    def test_build_sde_solver(self) -> None:
        """
        构建SDESolver算法实例成功
        """
        cfg = _make_full_config(implementation="sde_solver")
        algo = build_diffusion(cfg["diffusion"])
        self.assertIsInstance(algo, torch.nn.Module)

    def test_build_dpm_solver(self) -> None:
        """
        构建DPMSolver算法实例成功
        """
        cfg = _make_full_config(implementation="dpm_solver")
        algo = build_diffusion(cfg["diffusion"])
        self.assertIsInstance(algo, torch.nn.Module)

    def test_unsupported_implementation(self) -> None:
        """
        不支持的算法类型快速失败并抛出ValueError
        """
        bad_config: Dict[str, Any] = {
            "implementation": "invalid_algo",
            "timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        }
        with self.assertRaises(ValueError) as ctx:
            build_diffusion(bad_config)
        self.assertIn("不支持的", str(ctx.exception))

    def test_missing_implementation_field(self) -> None:
        """
        缺少implementation字段时抛出ValueError
        """
        bad_config: Dict[str, Any] = {
            "timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
        }
        with self.assertRaises(ValueError) as ctx:
            build_diffusion(bad_config)
        self.assertIn("implementation", str(ctx.exception))

    def test_missing_required_diffusion_keys(self) -> None:
        """
        diffusion配置缺少timesteps等必要字段时抛出ValueError
        """
        bad_config: Dict[str, Any] = {
            "implementation": "ddpm",
        }
        with self.assertRaises(ValueError) as ctx:
            build_diffusion(bad_config)
        self.assertIn("缺少必要字段", str(ctx.exception))


class TestConditionMode(unittest.TestCase):
    """
    条件模式相关测试
    """

    def test_condition_none(self) -> None:
        """
        condition.type为none时模型可正常构建和前向推理
        """
        cfg = _make_full_config(condition_type="none")
        model = build_model(cfg["model"])
        x = torch.randn(1, 1, 32, 32)
        t = torch.rand(1)
        out = model(x, t, None)
        self.assertEqual(out.shape, x.shape)

    def test_condition_class(self) -> None:
        """
        condition.type为class时模型可正常构建和前向推理
        """
        cfg = _make_full_config(condition_type="class", num_classes=10)
        model = build_model(cfg["model"])
        x = torch.randn(1, 1, 32, 32)
        t = torch.rand(1)
        condition = torch.randint(0, 10, (1,))
        out = model(x, t, condition)
        self.assertEqual(out.shape, x.shape)


class TestParameterPassing(unittest.TestCase):
    """
    参数传递验证测试
    """

    def test_num_classes_passed(self) -> None:
        """
        condition.num_classes正确传递，模型可接受对应范围的类别标签
        """
        cfg = _make_full_config(condition_type="class", num_classes=20)
        model = build_model(cfg["model"])
        x = torch.randn(1, 1, 32, 32)
        t = torch.rand(1)
        condition = torch.randint(0, 20, (1,))
        out = model(x, t, condition)
        self.assertEqual(out.shape, x.shape)

    def test_embedding_dim_passed(self) -> None:
        """
        condition.embedding_dim正确传递，前向推理正常完成
        """
        cfg = _make_full_config(
            condition_type="class", embedding_dim=256
        )
        model = build_model(cfg["model"])
        x = torch.randn(1, 1, 32, 32)
        t = torch.rand(1)
        condition = torch.randint(0, 10, (1,))
        out = model(x, t, condition)
        self.assertEqual(out.shape, x.shape)


class TestBuildFromConfig(unittest.TestCase):
    """
    build_from_config组合构建测试
    """

    def test_missing_diffusion_config(self) -> None:
        """
        缺少diffusion子配置时抛出ValueError
        """
        with self.assertRaises(ValueError) as ctx:
            build_from_config({"model": {"type": "unet", "in_channels": 1, "out_channels": 1, "condition": {"type": "none"}}})
        self.assertIn("diffusion", str(ctx.exception))

    def test_missing_model_config(self) -> None:
        """
        缺少model子配置时抛出ValueError
        """
        with self.assertRaises(ValueError) as ctx:
            build_from_config({
                "diffusion": {
                    "implementation": "ddpm",
                    "timesteps": 1000,
                    "beta_start": 0.0001,
                    "beta_end": 0.02,
                }
            })
        self.assertIn("model", str(ctx.exception))

    def test_returns_three_elements(self) -> None:
        """
        build_from_config返回三元组且元素类型正确
        """
        cfg = _make_full_config()
        result = build_from_config(cfg)
        self.assertEqual(len(result), 3)
        diffusion_inst, model_inst, config_ref = result
        self.assertIsInstance(diffusion_inst, torch.nn.Module)
        self.assertIsInstance(model_inst, torch.nn.Module)
        self.assertIsInstance(config_ref, dict)


class TestAllCombinations(unittest.TestCase):
    """
    4算法x2模型x2条件的16种组合全量构建测试
    """

    _IMPLEMENTATIONS = ["ddpm", "ddim", "sde_solver", "dpm_solver"]
    _MODEL_TYPES = ["unet", "dit"]
    _CONDITION_TYPES = ["none", "class"]

    def test_all_16_combinations(self) -> None:
        """
        16种算法x模型x条件组合均可成功构建并完成前向推理
        """
        passed = 0
        failed_combos = []

        for impl in self._IMPLEMENTATIONS:
            for model_type in self._MODEL_TYPES:
                for condition_type in self._CONDITION_TYPES:
                    combo_name = f"{impl}/{model_type}/{condition_type}"
                    try:
                        cfg = _make_full_config(
                            implementation=impl,
                            model_type=model_type,
                            condition_type=condition_type,
                        )
                        diffusion_inst, model_inst, _ = build_from_config(
                            cfg
                        )
                        # 前向推理验证
                        x = torch.randn(1, 1, 32, 32)
                        t = torch.rand(1)
                        condition = (
                            torch.randint(0, 10, (1,))
                            if condition_type == "class"
                            else None
                        )
                        out = model_inst(x, t, condition)
                        self.assertEqual(
                            out.shape, x.shape,
                            msg=f"输出形状不匹配: {combo_name}",
                        )
                        passed += 1
                    except Exception as e:
                        failed_combos.append(f"{combo_name}: {e}")

        self.assertEqual(
            failed_combos, [],
            msg=f"以下组合失败:\n" + "\n".join(failed_combos),
        )
        self.assertEqual(passed, 16)


class TestGetDevice(unittest.TestCase):
    """
    get_device工具函数测试
    """

    def test_default_device(self) -> None:
        """
        未指定device时默认返回cuda
        """
        device = get_device({"global": {}})
        self.assertEqual(device, torch.device("cuda"))

    def test_explicit_cpu(self) -> None:
        """
        显式指定cpu时返回cpu设备
        """
        device = get_device({"global": {"device": "cpu"}})
        self.assertEqual(device, torch.device("cpu"))

    def test_no_global_section(self) -> None:
        """
        配置中无global段时默认返回cuda
        """
        device = get_device({})
        self.assertEqual(device, torch.device("cuda"))


class TestNegativeCases(unittest.TestCase):
    """
    反例测试：验证工厂函数对非法配置的快速失败行为
    """

    def test_invalid_condition_type(self) -> None:
        """
        condition.type传入非法值时应抛出ValueError
        """
        config = _make_full_config("ddpm", "unet", "class")
        config["model"]["condition"]["type"] = "text"
        with self.assertRaises(ValueError) as ctx:
            build_model(config["model"])
        self.assertIn("condition.type", str(ctx.exception))

    def test_missing_in_channels(self) -> None:
        """
        缺少in_channels时应抛出ValueError而非KeyError
        """
        config = _make_full_config("ddpm", "unet", "none")
        del config["model"]["in_channels"]
        with self.assertRaises(ValueError) as ctx:
            build_model(config["model"])
        self.assertIn("必要字段", str(ctx.exception))

    def test_missing_out_channels(self) -> None:
        """
        缺少out_channels时应抛出ValueError而非KeyError
        """
        config = _make_full_config("ddpm", "unet", "none")
        del config["model"]["out_channels"]
        with self.assertRaises(ValueError) as ctx:
            build_model(config["model"])
        self.assertIn("必要字段", str(ctx.exception))

    def test_out_channels_not_equal_in_channels(self) -> None:
        """
        out_channels与in_channels不一致时应抛出ValueError
        """
        config = _make_full_config("ddpm", "unet", "none")
        config["model"]["out_channels"] = 3
        with self.assertRaises(ValueError) as ctx:
            build_model(config["model"])
        self.assertIn("一致", str(ctx.exception))

    def test_dpm_solver_order_3(self) -> None:
        """
        dpm_solver.order传入3时应抛出ValueError
        """
        config = _make_full_config("dpm_solver", "unet", "none")
        config["diffusion"]["dpm_solver"]["order"] = 3
        with self.assertRaises(ValueError) as ctx:
            build_diffusion(config["diffusion"])
        self.assertIn("order", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
