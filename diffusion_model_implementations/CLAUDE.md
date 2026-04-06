[根目录](../../CLAUDE.md) > **diffusion_model_implementations**

# diffusion_model_implementations - 即插即用的扩散模型框架

**最后更新时间**: 2026-04-06T20:36:06

---

## 变更日志 (Changelog)

### 2026-04-06
- 重新生成模块文档（原文件已删除后重建）
- 反映文件系统实际状态：所有源代码文件完好存在于磁盘

---

## 模块职责

本模块是一个**教学示例项目**，用于演示 Claude Code 的完整开发工作流。它实现了一个即插即用的扩散模型框架，核心设计目标是：**切换扩散算法或模型架构时，只改配置，不改业务代码**。

---

## 入口和启动

### 主入口

- **`runtime.py`** -- 配置解析与工厂入口，提供三个核心函数：
  - `load_config(config_path)` -- 读取 YAML 配置文件，返回配置字典
  - `build_from_config(config)` -- 根据配置构建扩散算法实例与噪声预测模型实例，返回三元组 `(diffusion_instance, model_instance, config)`
  - `get_device(config)` -- 从配置中提取设备信息，返回 `torch.device` 对象

### 运行方式

```bash
# 使用默认 config.yaml 运行全量组合测试
python runtime.py

# 运行单元测试
python -m pytest tests/ -v
# 或
python -m unittest tests.test_runtime_config -v
```

---

## 外部接口

### 扩散算法统一接口

所有扩散算法均继承 `torch.nn.Module`，实现以下三个方法：

```python
def forward_sample(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
    """前向加噪：从x_0直接采样到x_t"""

def reverse_sample(self, x_t: torch.Tensor, t: int, condition: Optional[torch.Tensor], model: nn.Module) -> torch.Tensor:
    """单步反向去噪：从x_t预测x_{t-1}"""

def reverse_sample_loop(self, shape: Tuple[int, ...], condition: Optional[torch.Tensor], model: nn.Module) -> torch.Tensor:
    """完整反向采样循环：从纯噪声生成x_0"""
```

### 噪声预测模型统一接口

所有模型均继承 `torch.nn.Module`，实现：

```python
def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor]) -> torch.Tensor:
    """预测噪声，输出形状与x_t一致"""
```

### 工厂函数

- **`diffusion_implementations.build_diffusion(config)`** -- 根据 `config["implementation"]` 构建扩散算法实例，支持 `"ddpm"` / `"ddim"` / `"sde_solver"` / `"dpm_solver"`
- **`models.build_model(model_config)`** -- 根据 `config["type"]` 构建模型实例，支持 `"unet"` / `"dit"`

---

## 关键依赖和配置

### 依赖

- Python 3.10+
- PyTorch >= 1.13
- PyYAML

### 配置文件 (`config.yaml`)

配置结构分为三个顶层键：

| 键 | 说明 |
|---|------|
| `diffusion` | 扩散算法配置：implementation、timesteps、beta_start、beta_end、beta_schedule、算法特定参数 |
| `model` | 模型配置：type、in_channels、out_channels、condition（type/num_classes/embedding_dim） |
| `global` | 全局配置：device、training（batch_size/learning_rate/num_epochs/gradient_clip） |

---

## 数据模型

### 支持的扩散算法

| 算法 | 类名 | 特点 | 额外参数 |
|------|------|------|---------|
| DDPM | `DDPM` | 标准扩散概率模型，前向加噪+反向去噪 | -- |
| DDIM | `DDIM` | 确定性/随机性子序列采样器，支持少步采样 | `eta` (0.0~1.0), `steps` |
| SDE Solver | `SDESolver` | 连续时间扩散采样器，Euler-Maruyama 离散化 | -- |
| DPM-Solver | `DPMSolver` | 多步加速采样器，支持一阶/二阶求解 | `order` (1/2), `steps` |

### 支持的模型架构

| 模型 | 类名 | 特点 | 可配置参数 |
|------|------|------|-----------|
| UNet | `UNet` | 编码器-瓶颈-解码器结构，含 skip connection 和自注意力 | base_channels, channel_multipliers, num_res_blocks |
| DiT | `DiT` | Diffusion Transformer，含 PatchEmbedding 和 AdaLN 调制 | patch_size, embed_dim, depth, num_heads, mlp_ratio |

### 条件模式

| 模式 | condition 值 | 说明 |
|------|-------------|------|
| 无条件 | `"none"` | condition 传 None |
| 类别条件 | `"class"` | condition 为类别索引张量 |

---

## 测试和质量

### 双层测试策略

1. **内置测试模块** -- 每个实现文件末尾包含 `if __name__ == "__main__"` 测试块：
   - 验证 forward_sample / reverse_sample / reverse_sample_loop 的接口形状
   - 覆盖 linear 和 cosine 两种 beta_schedule
   - 覆盖 class 和 none 两种条件模式

2. **单元测试** -- `tests/test_runtime_config.py`（unittest）：
   - `TestLoadConfig` -- 配置加载、文件不存在、缺少必要键、空配置
   - `TestBuildModel` -- unet/dit 构建、不支持的类型、通道数传递
   - `TestBuildDiffusion` -- ddpm/ddim/sde_solver/dpm_solver 构建、不支持的算法
   - `TestConditionMode` -- none/class 条件模式前向推理
   - `TestParameterPassing` -- num_classes、embedding_dim 参数传递
   - `TestBuildFromConfig` -- 组合构建、缺少子配置、返回三元组
   - `TestAllCombinations` -- 4 算法 x 2 模型 x 2 条件 = 16 种组合全量构建+前向推理
   - `TestGetDevice` -- 默认设备、显式 CPU、无 global 段
   - `TestNegativeCases` -- 反例快速失败测试

---

## 常见问题 (FAQ)

**Q: 如何切换扩散算法？**
A: 修改 `config.yaml` 中的 `diffusion.implementation` 字段为 `"ddpm"` / `"ddim"` / `"sde_solver"` / `"dpm_solver"`。

**Q: 如何切换模型架构？**
A: 修改 `config.yaml` 中的 `model.type` 字段为 `"unet"` 或 `"dit"`。

**Q: 如何启用条件生成？**
A: 将 `model.condition.type` 设为 `"class"` 并设置 `num_classes` 和 `embedding_dim`。

**Q: UNet 是否还包含 SimpleUNet？**
A: 不包含。当前 `unet.py` 只包含完整的 `UNet` 类（含 SinusoidalTimeEmbedding、_ResBlock、_AttentionBlock），不再有 SimpleUNet。

---

## 相关文件列表

```
diffusion_model_implementations/
├── runtime.py                                    # 配置解析与工厂入口
├── config.yaml                                   # 项目配置文件
├── docs/
│   ├── idea.md                                   # 项目需求与接口契约
│   └── coding_standards.md                       # 编码规范执行标准
├── tests/
│   ├── __init__.py                               # 测试包初始化（空文件）
│   └── test_runtime_config.py                    # 单元测试（16种组合全量验证）
├── diffusion_implementations/
│   ├── __init__.py                               # build_diffusion 工厂函数
│   ├── ddpm.py                                   # DDPM 算法
│   ├── ddim.py                                   # DDIM 算法
│   ├── sde_solver.py                             # SDE Solver
│   └── dpm_solver.py                             # DPM-Solver
└── models/
    ├── __init__.py                               # build_model 工厂函数
    ├── unet.py                                   # UNet（完整实现）
    └── dit.py                                    # DiT（Diffusion Transformer）
```
