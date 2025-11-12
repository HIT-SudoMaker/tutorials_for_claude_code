# diffusion_implementations - 扩散算法实现模块

[根目录](../CLAUDE.md) > **diffusion_implementations**

**最后更新时间**: 2025-11-12T19:06:30+08:00

---

## 变更日志 (Changelog)

### 2025-11-12
- 创建模块设计文档
- 完成四种扩散算法的统一接口实现
- 所有算法通过独立测试验证

---

## 模块职责

**diffusion_implementations** 模块提供四种主流扩散模型算法的统一接口实现：

1. **DDPM** - 基于 Markov 链的经典去噪扩散概率模型
2. **DDIM** - 非 Markov 过程，支持跳步加速采样
3. **SDESolver** - 基于随机微分方程的连续时间建模
4. **DPMSolver** - 快速高阶 ODE 求解器（10-20 步采样）

### 核心功能

- 统一的三方法接口：`forward_sample`, `reverse_sample`, `reverse_sample_loop`
- 支持线性和余弦 beta 调度
- 完整的类型提示和文档字符串
- 独立的测试代码块验证功能正确性

---

## 入口与启动

### 模块状态

已完成 - 所有四种算法已实现并通过测试

### 文件结构

```
diffusion_implementations/
├── __init__.py          # 模块初始化，导出四种算法类
├── ddpm.py             # DDPM 算法实现（451 行，含测试）
├── ddim.py             # DDIM 算法实现（476 行，含测试）
├── sde_solver.py       # SDE Solver 算法实现（171 行，含测试）
└── dpm_solver.py       # DPM-Solver 算法实现（456 行，含测试）
```

### 使用示例

```python
# 导入所有算法
from diffusion_implementations import DDPM, DDIM, SDESolver, DPMSolver
from models import SimpleUNet
import torch

# 初始化 DDPM
ddpm = DDPM(
    n_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule='linear',  # 'linear' 或 'cosine'
    device='cuda'
)

# 创建噪声预测模型
model = SimpleUNet(in_channels=1, out_channels=1).cuda()

# 前向加噪
x_0 = torch.randn(4, 1, 28, 28).cuda()
t = torch.randint(0, 1000, (4,)).cuda()
x_t, noise = ddpm.forward_sample(x_0, t)

# 单步反向去噪
condition = torch.randn(4, 1, 28, 28).cuda()
x_t_minus_1 = ddpm.reverse_sample(x_t, t, condition, model)

# 完整采样循环
x_T = torch.randn(4, 1, 28, 28).cuda()
x_0_generated = ddpm.reverse_sample_loop(x_T, condition, model)

# 使用 DDIM 加速采样（50 步替代 1000 步）
ddim = DDIM(n_timesteps=1000, beta_start=0.0001, beta_end=0.02, eta=0.0)
x_0_fast = ddim.reverse_sample_loop(x_T, condition, model, skip_steps=50)

# 使用 DPM-Solver 超快速采样（仅 20 步）
dpm = DPMSolver(n_timesteps=1000, beta_start=0.0001, beta_end=0.02, solver_order=2)
x_0_ultrafast = dpm.reverse_sample_loop(x_T, condition, model, fast_steps=20)
```

---

## 外部接口

### 统一接口规范

所有扩散算法类（`DDPM`, `DDIM`, `SDESolver`, `DPMSolver`）都继承 `torch.nn.Module`，并实现以下三个核心方法：

#### 1. `forward_sample`

```python
def forward_sample(
    self,
    x_0: torch.Tensor,
    t: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    前向加噪过程: x_0 -> x_t。

    Args:
        x_0 (torch.Tensor): 原始无噪声数据。
            shape: (batch_size, channels, height, width)
        t (torch.Tensor): 时间步索引（0 到 T-1）。
            shape: (batch_size,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - x_t: 加噪后的数据，shape 与 x_0 相同
            - noise: 采样的标准高斯噪声，shape 与 x_0 相同
    """
```

#### 2. `reverse_sample`

```python
def reverse_sample(
    self,
    x_t: torch.Tensor,
    t: torch.Tensor,
    condition: torch.Tensor,
    model: nn.Module
) -> torch.Tensor:
    """
    单步反向去噪: x_t -> x_{t-1}。

    Args:
        x_t (torch.Tensor): 当前时间步的带噪数据。
            shape: (batch_size, channels, height, width)
        t (torch.Tensor): 当前时间步索引。
            shape: (batch_size,)
        condition (torch.Tensor): 条件张量（用于条件生成）。
            shape: (batch_size, channels, height, width) 或兼容形状
        model (nn.Module): 噪声预测模型（如 U-Net）。
            输入 (x_t, t, condition)，输出预测噪声

    Returns:
        torch.Tensor: 去噪一步后的数据 x_{t-1}，shape 与 x_t 相同
    """
```

#### 3. `reverse_sample_loop`

```python
def reverse_sample_loop(
    self,
    x_T: torch.Tensor,
    condition: torch.Tensor,
    model: nn.Module
) -> torch.Tensor:
    """
    完整反向采样循环: x_T -> x_0。

    Args:
        x_T (torch.Tensor): 初始纯高斯噪声（标准正态分布采样）。
            shape: (batch_size, channels, height, width)
        condition (torch.Tensor): 条件张量（用于条件生成）。
            shape 需与模型输入兼容
        model (nn.Module): 噪声预测模型

    Returns:
        torch.Tensor: 生成的数据 x_0，shape 与 x_T 相同
    """
```

---

## 算法对比

### 采样速度与质量对比

| 算法 | 标准步数 | 最小步数 | 相对速度 | 确定性 | 生成质量 |
|-----|---------|---------|---------|--------|---------|
| **DDPM** | 1000 | 1000 | 1x（基准） | 否（随机） | 高 |
| **DDIM** | 1000 | 50 | 20x | 是（eta=0） | 高 |
| **SDE Solver** | 1000 | 100 | 10x | 否（随机） | 高 |
| **DPM-Solver** | 1000 | 10-20 | 50-100x | 是 | 高 |

### 算法选择指南

- **追求最高质量**：DDPM（1000 步完整采样）
- **平衡速度与质量**：DDIM（50 步跳步采样）
- **超快速采样**：DPM-Solver（10-20 步）
- **连续时间建模**：SDE Solver（理论研究）

---

## 算法详解

### 1. DDPM (ddpm.py)

**核心原理**：
- 基于 Markov 链，逐步添加高斯噪声将数据转为标准正态分布
- 训练模型预测每一步添加的噪声
- 反向过程通过逐步去噪生成数据

**关键公式**：
```
前向过程: x_t = √α_bar_t * x_0 + √(1-α_bar_t) * ε
反向过程: x_{t-1} = 1/√α_t * (x_t - β_t/√(1-α_bar_t) * ε_θ) + σ_t * z
```

**参数说明**：
- `beta_schedule`: 支持 'linear' 和 'cosine' 两种调度
- `beta_start/beta_end`: 噪声水平范围（典型值：0.0001 ~ 0.02）

**测试覆盖**：
- 线性和余弦调度的初始化
- 前向加噪的正确性
- 单步反向去噪
- 完整采样循环（测试用 50 步）

### 2. DDIM (ddim.py)

**核心原理**：
- 使用非 Markov 前向过程，允许跳步采样
- 训练与 DDPM 相同，但采样更高效
- 通过 eta 参数控制随机性（0=确定性，1=DDPM）

**关键公式**：
```
x_{t-1} = √α_bar_{t-1} * predicted_x_0
          + √(1 - α_bar_{t-1} - σ_t²) * ε_θ
          + σ_t * z
```

**参数说明**：
- `eta`: 随机性参数（0.0=完全确定性，1.0=DDPM 等价）
- `skip_steps`: 跳步采样的实际步数（如 50 步替代 1000 步）

**特色功能**：
- 确定性采样（eta=0）：相同输入总是产生相同输出
- 可重复性：适合需要精确控制的场景
- 跳步采样：20 倍速度提升

### 3. SDE Solver (sde_solver.py)

**核心原理**：
- 使用随机微分方程（SDE）建模扩散过程
- 前向过程为随机游走，反向过程求解反向 SDE
- 支持 VP-SDE（Variance Preserving SDE）

**关键公式**：
```
前向 SDE: dx = -0.5 * β_t * x dt + √β_t dw
反向 SDE: dx = [-0.5 * β_t * x - β_t * ∇log p(x)] dt + √β_t dw
```

**参数说明**：
- `sde_type`: 当前仅支持 'vpsde'
- 使用 Euler-Maruyama 方法进行数值求解

**适用场景**：
- 连续时间建模
- 理论研究和分析
- Score-based 生成模型

### 4. DPM-Solver (dpm_solver.py)

**核心原理**：
- 为扩散 ODE 设计的快速高阶求解器
- 通过数值方法（类似 Runge-Kutta）求解积分
- 保证收敛阶，仅需 10-20 步即可高质量采样

**关键公式**：
```
扩散 ODE: dx/dλ = (x - x_θ(x, λ)) / σ(λ)
λ_t = log(α_t / σ_t) = 0.5 * log(α_bar_t / (1 - α_bar_t))
```

**参数说明**：
- `solver_order`: 求解器阶数（1/2/3），阶数越高质量越好
- `fast_steps`: 采样步数（典型值：10-20 步）

**性能特点**：
- 1 阶（Euler）：最快，质量稍低
- 2 阶：推荐，平衡速度与质量
- 3 阶：最高质量，略慢

---

## 关键依赖与配置

### 依赖项

- **PyTorch**: 深度学习框架（>= 1.13）
- **NumPy**: 数值计算（用于时间步生成）
- **Math**: 标准库（用于余弦调度）

### 配置参数（来自 config.yaml）

```yaml
diffusion:
  implementation: 'ddpm'  # 'ddpm' | 'ddim' | 'sde_solver' | 'dpm_solver'
  timesteps: 1000         # 训练时的总时间步数
  beta_start: 0.0001      # 初始噪声水平
  beta_end: 0.02          # 最终噪声水平
  beta_schedule: 'linear' # 'linear' | 'cosine'

  # DDIM 专用参数
  ddim:
    eta: 0.0              # 随机性控制（0=确定性，1=DDPM）
    skip_steps: 50        # 跳步采样步数

  # DPM-Solver 专用参数
  dpm_solver:
    solver_order: 2       # 求解器阶数（1/2/3）
    prediction_type: 'epsilon'  # 'epsilon' / 'sample'

global:
  device: 'cuda'          # 'cuda' | 'cpu'
```

---

## 数据模型

### 输入输出规范

| 参数 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `x_0` | `[B, C, H, W]` | torch.Tensor | 原始无噪声数据 |
| `x_t` | `[B, C, H, W]` | torch.Tensor | 时间步 t 的带噪数据 |
| `x_T` | `[B, C, H, W]` | torch.Tensor | 初始纯高斯噪声 |
| `t` | `[B]` | torch.Tensor (long) | 时间步索引（0 到 T-1） |
| `condition` | `[B, C, H, W]` | torch.Tensor | 条件信息（可选） |
| `noise` | `[B, C, H, W]` | torch.Tensor | 标准高斯噪声 |

### 预计算系数（存储在类属性中）

| 系数 | 形状 | 说明 |
|-----|------|------|
| `beta` | `[T]` | Beta 调度序列 |
| `alpha` | `[T]` | Alpha 序列（1 - beta） |
| `alpha_bar` | `[T]` | Alpha 累积乘积 |
| `sqrt_alpha_bar` | `[T]` | √α_bar（前向过程） |
| `sqrt_one_minus_alpha_bar` | `[T]` | √(1-α_bar)（前向过程） |
| `sqrt_recip_alpha` | `[T]` | 1/√α（反向过程，仅 DDPM） |
| `beta_tilde` | `[T]` | 后验方差（反向过程，仅 DDPM） |
| `lambda_t` | `[T]` | λ_t（DPM-Solver 专用） |

---

## 测试与质量

### 测试策略

每个算法文件包含完整的测试模块（`if __name__ == "__main__"`），覆盖以下内容：

#### DDPM 测试（ddpm.py）
1. 初始化测试（线性和余弦调度）
2. 前向加噪过程验证
3. 单步反向去噪验证
4. 完整采样循环（50 步快速测试）
5. Beta 范围和数值稳定性检查

#### DDIM 测试（ddim.py）
1. 不同 eta 值的初始化（0.0, 0.5, 1.0）
2. 前向加噪（与 DDPM 相同）
3. 单步反向去噪（标准步长）
4. 跳步采样（1000 步 -> 50 步）
5. 确定性采样的可重复性验证

#### SDE Solver 测试（sde_solver.py）
1. VP-SDE 初始化
2. 前向加噪过程
3. 单步反向 SDE 更新
4. 完整采样循环（100 步）

#### DPM-Solver 测试（dpm_solver.py）
1. 不同阶数的初始化（1/2/3 阶）
2. 前向加噪过程
3. x_0 预测功能
4. 1 阶和 2 阶更新验证
5. 完整快速采样循环（20 步）
6. λ_t 使用验证

### 运行测试

```bash
# 测试所有算法
python diffusion_implementations/ddpm.py
python diffusion_implementations/ddim.py
python diffusion_implementations/sde_solver.py
python diffusion_implementations/dpm_solver.py

# Windows 批量测试
for %f in (diffusion_implementations\*.py) do python %f
```

### 质量检查清单

- [x] 所有算法通过独立测试
- [x] 输出形状与输入一致
- [x] 设备兼容（CPU/CUDA）
- [x] 边界条件处理（t=0, t=T-1）
- [x] 数值稳定性（防止除零、溢出）
- [x] 类型提示完整
- [x] 文档字符串符合 Google 风格
- [x] 变量命名遵循规范（避免缩写）
- [x] 代码符合 PEP 8 标准

---

## 常见问题 (FAQ)

### Q1: 如何选择 beta_schedule？

**A**:
- **Linear**: 原始 DDPM 论文使用，简单直接
- **Cosine**: Improved DDPM 提出，生成质量更好，推荐用于图像生成

### Q2: DDIM 的 eta 参数如何设置？

**A**:
- **eta=0.0**: 完全确定性采样，相同输入产生相同输出，适合需要可重复性的场景
- **eta=1.0**: 等价于 DDPM，完全随机，多样性最高
- **0 < eta < 1**: 半确定性，平衡多样性和可控性

### Q3: 为什么 DPM-Solver 这么快？

**A**: DPM-Solver 使用高阶 ODE 求解器，通过数值方法（类似 Runge-Kutta）精确求解扩散 ODE，避免了逐步迭代的低效。2 阶求解器仅需 20 步即可达到 DDPM 1000 步的质量。

### Q4: 四种算法的训练过程有何不同？

**A**: **训练过程完全相同**！所有算法都使用相同的损失函数（MSE 噪声预测损失），只是采样策略不同：
- DDPM: 逐步采样 1000 步
- DDIM: 跳步采样 50 步
- SDE Solver: SDE 反向求解
- DPM-Solver: ODE 高阶求解

### Q5: 如何处理 GPU 内存不足？

**A**:
- 减小 `batch_size`
- 使用 DDIM 或 DPM-Solver 的快速采样（减少时间步）
- 使用 `torch.cuda.amp`（自动混合精度）
- 降低图像分辨率

### Q6: condition 参数如何使用？

**A**:
- **无条件生成**: 传入零张量 `torch.zeros_like(x_t)`
- **类别条件**: 通过嵌入层将类别标签转为张量
- **图像条件**: 直接传入参考图像或通过编码器提取特征
- **文本条件**: 使用 CLIP/T5 编码器提取文本特征

### Q7: 为什么测试时使用 SimpleUNet 而不是完整 UNet？

**A**: SimpleUNet 是一个简化的占位符模型，仅用于验证扩散算法接口的正确性。实际训练和生成需要使用完整的 UNet 模型（包含时间嵌入、注意力、残差块）。

### Q8: 如何验证算法实现的正确性？

**A**:
1. 检查输出形状是否与输入一致
2. 验证前向加噪后的数据逐渐接近标准高斯分布
3. 验证反向采样能生成合理的数据分布
4. 对比论文中的数值实验结果
5. 可视化采样过程的中间状态

---

## 相关文件列表

### 实现文件

- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\diffusion_implementations\__init__.py` - 模块初始化（22 行）
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\diffusion_implementations\ddpm.py` - DDPM 实现（451 行）
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\diffusion_implementations\ddim.py` - DDIM 实现（476 行）
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\diffusion_implementations\sde_solver.py` - SDE Solver 实现（171 行）
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\diffusion_implementations\dpm_solver.py` - DPM-Solver 实现（456 行）

### 相关文档

- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\idea.md` - 项目需求和算法原理说明
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\coding_paradigm.md` - Python 编码规范
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\config.yaml` - 全局配置文件

### 依赖模块

- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\models\unet.py` - 噪声预测模型

---

## 参考资源

### 核心论文

1. **DDPM** - Ho, J., Jain, A., & Abbeel, P. (2020). "Denoising diffusion probabilistic models." NeurIPS 2020.
2. **DDIM** - Song, J., Meng, C., & Ermon, S. (2021). "Denoising diffusion implicit models." ICLR 2021.
3. **Score-based SDE** - Song, Y., et al. (2021). "Score-based generative modeling through stochastic differential equations." ICLR 2021.
4. **DPM-Solver** - Lu, C., et al. (2022). "DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling." NeurIPS 2022.

### 实现参考

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - 生产级扩散模型库
- [OpenAI Guided Diffusion](https://github.com/openai/guided-diffusion) - DDPM 官方实现
- [DPM-Solver GitHub](https://github.com/LuChengTHU/dpm-solver) - DPM-Solver 官方实现

### 教程资源

- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) - 带注释的 DDPM 实现
- [Lil'Log: Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - 扩散模型理论详解

---

**下一步行动**:
1. 完成 `models/unet.py` 中完整 UNet 的实现
2. 开发基于配置文件的算法工厂类
3. 实现训练循环和采样脚本
4. 在 MNIST 数据集上验证端到端训练流程
