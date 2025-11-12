# 即插即用的多种扩散模型实现

**核心目标:**

您好！请根据以下详细要求，为我创建一个用于“即插即用的多种扩散模型实现”的Python项目。此项目的核心是一个灵活的扩散模型框架，其关键特性是能够通过一个 `config.yaml` 配置文件，轻松地切换和控制多种扩散算法（如 DDPM, DDIM, SDE Solver, DPM-Solver）。

请严格遵循下文定义的项目文件结构、变量命名和接口统一规范。编程风格请参考“coding_paradigm.md”中的PEP 8标准、命名约定、类型提示、文档字符串等规范，避免重复描述。

-----

## 1. 四种扩散模型的具体原理

为确保实现正确，请参考以下原理描述：

- **DDPM (Denoising Diffusion Probabilistic Models)**:  
  基于非平衡热力学，通过Markov链逐步添加高斯噪声将数据转化为已知先验分布（通常为标准高斯噪声）。逆向过程训练模型逐步去噪，从噪声生成数据。核心是学习噪声预测，实现渐进式采样。

- **DDIM (Denoising Diffusion Implicit Models)**:  
  DDPM的变体，使用非Markov前向过程，允许跳步采样以加速生成过程。训练与DDPM相同，但采样更高效，通过隐式模型减少步骤，同时保持生成质量。

- **SDE Solver (Stochastic Differential Equation Solver)**:  
  使用随机微分方程（SDE）模拟扩散过程，将数据平滑转换为噪声分布。前向过程为随机游走，逆向过程求解SDE的反向方程，从噪声生成数据，支持连续时间建模。

- **DPM-Solver (Diffusion Probabilistic Model Solver)**:  
  为扩散ODE设计的快速高阶求解器，通过数值方法（如Runge-Kutta）求解扩散方程的积分。保证收敛阶，仅需少量步骤（10-20步）即可采样，相比传统方法显著加速。

-----

## 2. 项目目录结构

请首先创建如下所示的目录与文件结构：

```
项目根目录/
├── config.yaml
├── diffusion_implementations/
│   ├── __init__.py
│   ├── ddpm.py
│   ├── ddim.py
│   ├── sde_solver.py
│   └── dpm_solver.py
└── models/
    ├── __init__.py
    └── unet.py
```

-----

## 3. 核心组件详细规格

### A. `config.yaml` 配置文件

该文件用于集中管理和控制所有关键参数。

**`config.yaml` 示例内容:**

```yaml
# -----------------------------
# 扩散模型实现与参数配置
# -----------------------------
diffusion:
  # 可选值: 'ddpm', 'ddim', 'sde_solver', 'dpm_solver'
  implementation: 'ddpm' 
  timesteps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  
# -----------------------------
# 噪声预测模型 (U-Net) 配置
# -----------------------------
model:
  in_channels: 1
  out_channels: 1
  # ... 此处可定义其他U-Net模型所需的参数，例如通道数、注意力头数等
  
# -----------------------------
# 全局与训练配置
# -----------------------------
global:
  device: 'cuda' # 可选 'cpu'
  # ... 此处可定义训练参数, 如 batch_size, learning_rate 等
```

### B. `diffusion_implementations/` 文件夹

此文件夹包含所有扩散算法的具体实现。其中的每一个Python文件（`ddpm.py`, `ddim.py` 等）都必须遵守以下严格规范：

**所有实现文件的通用要求:**

1.  **类的实现方式:** 每个文件中都必须定义一个核心类（例如 `DDPM`, `DDIM`），该类必须继承自 `torch.nn.Module`。

2.  **统一的方法命名:** 为了确保所有算法模块可以被无缝替换和调用，每个类都必须实现以下三个核心方法，且方法名、参数和返回格式必须高度统一（初始化方法可根据原理不同而有所区别）：

      * `forward_sample()`: 实现前向加噪过程，从 $x_0 \rightarrow x_t$。
      * `reverse_sample()`: 实现单步反向去噪过程，从 $x_t \rightarrow x_{t-1}$。
      * `reverse_sample_loop()`: 实现从纯噪声 $x_T$ 生成最终数据 $x_0$ 的完整反向采样循环。

3.  **标准化的变量命名:**

      * 使用 `x` 来表示数据张量。
      * 使用 `condition` 来表示指导生成过程的条件张量。
      * 使用 `x_0` 表示初始的、无噪声的数据。
      * 使用 `x_t` 表示在时间步 `t` 的带噪数据。
      * 使用 `x_T` 表示初始的纯高斯噪声。

4.  **规范的文档字符串 (Docstring):** 所有方法都必须包含清晰的文档字符串，并应统一采用 `Args:` 和 `Returns:` 的风格。

      * **文档字符串示例:**
        ```python
        def reverse_sample(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, model: nn.Module):
            """
            根据模型预测的噪声，执行单步反向去噪。

            Args:
                x_t (torch.Tensor): 当前时间步t的带噪数据。
                t (torch.Tensor): 当前的时间步。
                condition (torch.Tensor): 用于指导生成的条件张量。
                model (nn.Module): 用于预测噪声的神经网络模型。
            
            Returns:
                torch.Tensor: 去噪一步后得到的数据x_t-1。
            """
            # ... 具体实现代码 ...
        ```

5.  **独立的测试样例:** 每个文件的末尾都必须包含一个 `if __name__ == "__main__":` 测试模块。该模块需要：

      * 实例化对应的扩散实现类 (例如 `diffusion = DDPM(...)`)。
      * 创建一个能返回正确形状张量的“伪模型”（使用以下的 `SimpleUNet`），用于测试。
      * 创建用于测试的伪数据张量，包括数据 `x` 和条件 `condition`。
      * 分别调用并打印 `forward_sample()`, `reverse_sample()`, 和 `reverse_sample_loop()` 这三个核心方法的输出张量形状 (shape)，以验证其基本功能是否正常运行。

**SimpleUNet伪模型代码（用于测试）:**

```python
import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    """
    一个简单的U-Net模型，用于测试噪声预测。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 1, out_channels, kernel_size=3, padding=1)  # 简单卷积层模拟U-Net

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        预测噪声。

        Args:
            x_t (torch.Tensor): 带噪数据。
            t (torch.Tensor): 时间步。
            condition (torch.Tensor): 条件张量。
        
        Returns:
            torch.Tensor: 预测的噪声，与x_t形状相同。
        """
        # 模拟嵌入时间和条件
        t_emb = t.view(t.size(0), 1, 1, 1).expand(-1, 1, x_t.size(2), x_t.size(3))
        input = torch.cat([x_t, t_emb, condition], dim=1)  # 假设condition与x_t形状兼容
        return self.conv(input)
```

**不同算法文件的具体逻辑 (`ddpm.py`, `ddim.py` 等):**

  * 每个文件中的方法实现应严格遵循其对应的算法理论。
  * 每个类的 `__init__` 初始化方法接收 `n_timesteps`、`device` 等参数，并应提前计算好该算法所需的 `beta`、`alpha`、`alpha_bar` 等参数序列。

### C. `models/` 文件夹

此文件夹用于存放用于预测噪声的神经网络模型架构。

  * **`unet.py`:**  
      * 请在此文件中实现一个标准的U-Net模型类 (例如 `UNet`)，并使其继承自 `torch.nn.Module`。  
      * 该模型的 `forward` 方法必须接收三个输入：加噪后的数据 `x_t`、条件 `condition`、以及当前的时间步 `t`。  
      * 该方法应返回一个与输入 `x_t` 形状完全相同的张量，代表模型预测出的噪声。