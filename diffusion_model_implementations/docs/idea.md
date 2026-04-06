# AI临时需求卡：即插即用的扩散模型框架

## 1. 文档定位

本文件只面向当前模块开发，用于定义“这次要实现什么”。

- 本文件是临时需求卡，不是长期规范
- 编码风格与审查规则由 `coding_standards.md` 负责
- 若本轮用户要求与本文件冲突，以用户要求为准

## 2. 功能要求

- 通过 `runtime.py` 读取 `config.yaml` 并构建扩散算法与模型实例
- `diffusion.implementation` 必须支持：`ddpm`、`ddim`、`sde_solver`、`dpm_solver`
- `model.type` 必须支持：`unet`、`dit`
- `model.condition.type` 必须支持：`none`、`class`
- 切换扩散算法或模型时，只改配置，不改业务代码
- 对不支持的配置值，必须快速失败并抛出清晰异常

## 3. 接口约束

### 3.1 扩散算法接口

每个扩散算法核心类必须继承 `torch.nn.Module`，并实现以下方法签名：

```python
def forward_sample(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
def reverse_sample(self, x_t: torch.Tensor, t: int, condition: Optional[torch.Tensor], model: nn.Module) -> torch.Tensor:
def reverse_sample_loop(self, shape: Tuple[int, ...], condition: Optional[torch.Tensor], model: nn.Module) -> torch.Tensor:
```

**方法语义：**

- `forward_sample(x_0, t)`：前向加噪过程，返回 `x_t`。从 `x_0` 按噪声调度直接加噪到时间步 `t`，不涉及模型推理
- `reverse_sample(x_t, t, condition, model)`：单步反向去噪，返回 `x_{t-1}`。给定时间步 `t` 的带噪数据，调用模型预测噪声并去除，返回上一步的去噪结果
- `reverse_sample_loop(shape, condition, model)`：完整反向采样循环，返回生成的 `x_0`。从纯噪声 `x_T` 开始，逐步去噪直到得到最终样本

### 3.2 噪声预测模型接口

每个噪声预测模型类必须继承 `torch.nn.Module`，并实现以下方法签名：

```python
def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor]) -> torch.Tensor:
```

### 3.3 统一语义

- `x_0`：无噪声原始数据
- `x_t`：时间步 `t` 的带噪数据
- `x_{t-1}`：时间步 `t-1` 的去噪结果（`reverse_sample` 的返回值）
- `x_T`：纯高斯噪声（`t = T-1`，即总步数 `timesteps` 对应的最大索引）
- `t`：时间步索引，取值范围 `[0, T-1]`，其中 `T` 为配置中的 `timesteps`。在 `reverse_sample` 中，`t` 表示当前所在的时间步
- `condition`：条件张量，类型为 `Optional[torch.Tensor]`
- 模型 `forward` 的输出形状必须与 `x_t` 一致
- 当 `condition.type == "none"` 时，`condition` 传 `None`，模型内部直接跳过条件处理
- 当 `condition.type == "class"` 时，`condition` 为类别索引张量

## 4. 配置约束

AI 必须围绕以下配置结构实现，不得擅自更名核心键：

```yaml
diffusion:
  implementation: "ddpm"
  timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  beta_schedule: "linear"
  ddim:
    eta: 0.0           # DDIM 随机性控制：0.0 为完全确定性采样，1.0 退化为 DDPM
    steps: 50          # 采样步数，可小于训练 timesteps
  dpm_solver:
    order: 2           # 多步求解器阶数：1/2
    steps: 20          # 采样步数

model:
  type: "unet"
  in_channels: 1
  out_channels: 1
  unet: {}
  dit: {}
  condition:
    type: "class"
    num_classes: 10
    embedding_dim: 128

global:
  device: "cuda"
  training:
    batch_size: 16
    learning_rate: 0.0001
    num_epochs: 100
    gradient_clip: 1.0
```

## 5. 验收与非目标

满足以下条目即可视为完成：

- `runtime.py` 能根据配置构建支持的扩散算法与模型实例
- 所有扩散算法实现都满足第 3.1 节接口约束
- 所有模型实现都满足第 3.2 节接口约束
- `condition.type == "none"` 和 `condition.type == "class"` 两种模式都可构建
- 至少存在一种最小验证方式：文件内自测块、测试文件，或二者兼有

除非用户明确提出，否则以下内容不属于本卡默认范围：

- 完整训练器
- 数据集下载与预处理
- 文本条件生成
- classifier-free guidance
- 分布式训练
- 与当前骨架无关的大规模重构
