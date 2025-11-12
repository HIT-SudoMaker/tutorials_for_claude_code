# models - 神经网络模型模块

[根目录](../../CLAUDE.md) > **models**

**最后更新时间**: 2025-11-12T17:29:27+08:00

---

## 📋 变更日志

### 2025-11-12
- 创建模块设计文档
- 定义 U-Net 模型接口规范

---

## 🎯 模块职责

**models** 模块负责提供用于扩散模型的神经网络架构，主要用于噪声预测任务。

### 核心功能
- 实现 U-Net 架构，作为标准的噪声预测模型
- 接收带噪数据、时间步和条件信息
- 输出预测的噪声张量

---

## 🚀 入口与启动

### 模块状态
🚧 **计划中** - 尚未实现

### 计划的文件结构
```
models/
├── __init__.py          # 模块初始化，导出模型类
└── unet.py             # U-Net 噪声预测模型
```

### 使用示例（计划）
```python
from models import UNet
import torch

# 初始化模型
model = UNet(
    in_channels=1,      # 输入通道数（如灰度图为 1，RGB 为 3）
    out_channels=1,     # 输出通道数（通常与输入相同）
    base_channels=64,   # 基础通道数
    channel_multipliers=[1, 2, 4, 8],  # 各层通道倍数
    num_res_blocks=2,   # 每层的残差块数量
    attention_resolutions=[16, 8]  # 使用注意力机制的分辨率
)

# 前向传播
batch_size = 4
height, width = 64, 64
x_t = torch.randn(batch_size, 1, height, width)  # 带噪数据
t = torch.randint(0, 1000, (batch_size,))        # 时间步
condition = torch.randn(batch_size, 1, height, width)  # 条件（可选）

predicted_noise = model(x_t, t, condition)
print(f"预测噪声形状: {predicted_noise.shape}")  # 应与 x_t 形状相同
```

---

## 🔌 外部接口

### UNet 类接口规范

```python
class UNet(nn.Module):
    """
    U-Net 噪声预测模型。

    Args:
        in_channels (int): 输入图像通道数
        out_channels (int): 输出图像通道数（通常等于 in_channels）
        base_channels (int): 基础通道数，默认 64
        channel_multipliers (List[int]): 各层通道数倍增因子
        num_res_blocks (int): 每个分辨率层的残差块数量
        attention_resolutions (List[int]): 应用注意力机制的分辨率列表
        dropout (float): Dropout 概率，默认 0.0
    """

    def forward(self,
                x_t: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        预测噪声。

        Args:
            x_t (torch.Tensor): 带噪数据，形状 [batch_size, in_channels, height, width]
            t (torch.Tensor): 时间步，形状 [batch_size]
            condition (torch.Tensor): 条件张量，形状与 x_t 兼容或可广播

        Returns:
            torch.Tensor: 预测的噪声，形状与 x_t 相同
        """
```

### 关键组件

#### 1. 时间步嵌入 (Time Embedding)
```python
def time_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    将时间步 t 转换为高维嵌入向量。

    通常使用正弦位置编码（Sinusoidal Positional Encoding）：
    - 低频分量：捕捉全局时间信息
    - 高频分量：捕捉局部时间变化

    Args:
        t: 时间步，形状 [batch_size]
        dim: 嵌入维度

    Returns:
        时间嵌入，形状 [batch_size, dim]
    """
```

#### 2. 下采样路径 (Downsampling Path)
- 逐步降低空间分辨率
- 增加通道数
- 提取多尺度特征

#### 3. 瓶颈层 (Bottleneck)
- 最低分辨率的特征处理
- 通常包含自注意力机制

#### 4. 上采样路径 (Upsampling Path)
- 逐步恢复空间分辨率
- 通过跳跃连接 (skip connections) 融合下采样特征

#### 5. 残差块 (Residual Block)
```python
class ResidualBlock(nn.Module):
    """
    残差块，包含：
    - 分组归一化 (Group Normalization)
    - 激活函数 (SiLU/Swish)
    - 卷积层
    - 时间嵌入注入
    - 残差连接
    """
```

#### 6. 自注意力模块 (Self-Attention)
```python
class AttentionBlock(nn.Module):
    """
    多头自注意力模块，用于捕捉长距离依赖。
    通常应用于较低分辨率的特征图。
    """
```

---

## 📦 关键依赖与配置

### 依赖项
- **PyTorch**: 深度学习框架
- **torch.nn**: 神经网络模块

### 配置参数（来自 config.yaml）

```yaml
model:
  in_channels: 1              # 输入通道数
  out_channels: 1             # 输出通道数
  base_channels: 64           # 基础通道数
  channel_multipliers: [1, 2, 4, 8]  # 通道倍增因子
  num_res_blocks: 2           # 残差块数量
  attention_resolutions: [16, 8]  # 注意力层的分辨率
  dropout: 0.0                # Dropout 概率
```

### 模型架构示例

以 `base_channels=64`, `channel_multipliers=[1, 2, 4, 8]` 为例：

```
输入: [B, 1, 64, 64]

下采样路径:
  Level 0: [B, 64, 64, 64]   (64x1)
  Level 1: [B, 128, 32, 32]  (64x2) + Attention
  Level 2: [B, 256, 16, 16]  (64x4) + Attention
  Level 3: [B, 512, 8, 8]    (64x8)

瓶颈:
  [B, 512, 8, 8] + Attention

上采样路径:
  Level 3: [B, 512, 8, 8]    + 跳跃连接
  Level 2: [B, 256, 16, 16]  + 跳跃连接 + Attention
  Level 1: [B, 128, 32, 32]  + 跳跃连接 + Attention
  Level 0: [B, 64, 64, 64]   + 跳跃连接

输出: [B, 1, 64, 64]
```

---

## 📊 数据模型

### 输入输出规范

| 参数 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `x_t` | `[B, C, H, W]` | torch.Tensor | 带噪数据 |
| `t` | `[B]` | torch.Tensor (long) | 时间步索引 |
| `condition` | `[B, C, H, W]` | torch.Tensor | 条件信息（可选） |
| **返回值** | `[B, C, H, W]` | torch.Tensor | 预测噪声 |

### SimpleUNet 测试模型

用于快速测试的简化版本（来自 `idea.md`）：

```python
class SimpleUNet(nn.Module):
    """
    简化的 U-Net 模型，用于测试扩散算法。

    注意：此模型仅用于测试，不适合实际训练。
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels + 1, out_channels, kernel_size=3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        # 简单的时间嵌入（扩展标量到空间维度）
        t_emb = t.view(t.size(0), 1, 1, 1).expand(-1, 1, x_t.size(2), x_t.size(3))

        # 拼接输入（假设 condition 与 x_t 形状兼容）
        input_tensor = torch.cat([x_t, t_emb, condition], dim=1)

        return self.conv(input_tensor)
```

---

## 🧪 测试与质量

### 测试策略（计划）

`unet.py` 应包含以下测试代码：

```python
if __name__ == "__main__":
    import torch

    # 测试参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    in_channels = 1
    out_channels = 1
    height, width = 64, 64
    n_timesteps = 1000

    # 实例化模型
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=64,
        channel_multipliers=[1, 2, 4],
        num_res_blocks=2,
        attention_resolutions=[16]
    ).to(device)

    # 创建测试数据
    x_t = torch.randn(batch_size, in_channels, height, width).to(device)
    t = torch.randint(0, n_timesteps, (batch_size,)).to(device)
    condition = torch.randn(batch_size, in_channels, height, width).to(device)

    # 测试前向传播
    print("测试 UNet 前向传播:")
    print(f"  输入 x_t 形状: {x_t.shape}")
    print(f"  时间步 t 形状: {t.shape}")
    print(f"  条件 condition 形状: {condition.shape}")

    with torch.no_grad():
        predicted_noise = model(x_t, t, condition)

    print(f"  输出噪声形状: {predicted_noise.shape}")

    # 验证输出形状
    assert predicted_noise.shape == x_t.shape, "输出形状应与输入 x_t 相同"

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    print("\n✓ 所有测试通过！")
```

### 质量检查清单
- [ ] 输出形状与输入 `x_t` 完全一致
- [ ] 支持不同的输入分辨率
- [ ] 时间嵌入正确注入到网络中
- [ ] 跳跃连接正确实现
- [ ] 注意力模块在指定分辨率正常工作
- [ ] 代码符合 PEP 8 和 `coding_paradigm.md` 规范
- [ ] 所有方法包含类型提示和文档字符串

---

## ❓ 常见问题 (FAQ)

### Q1: 为什么使用 U-Net 架构？
**A**: U-Net 的优势包括：
1. **多尺度特征**: 下采样和上采样路径捕捉不同尺度信息
2. **跳跃连接**: 保留空间细节，避免信息丢失
3. **成熟稳定**: 在图像生成任务中验证有效
4. **灵活性**: 可轻松调整深度和宽度

### Q2: 时间步嵌入为什么重要？
**A**:
- 模型需要知道当前处于扩散过程的哪个阶段
- 不同时间步的噪声水平不同，需要不同的去噪策略
- 时间嵌入使模型能够学习时间相关的特征

### Q3: 何时使用注意力机制？
**A**:
- **低分辨率特征**: 通常在 16×16 或 8×8 分辨率
- **计算成本**: 注意力复杂度为 O(n²)，不适用于高分辨率
- **权衡**: 在性能和计算成本之间平衡

### Q4: `condition` 参数如何使用？
**A**:
- **类别条件**: 通过嵌入层转换为特征图
- **文本条件**: 使用 CLIP 或 T5 编码器提取特征
- **图像条件**: 直接拼接或通过交叉注意力融合
- **无条件**: 传入零张量或不使用

### Q5: SimpleUNet 和完整 UNet 的区别？
**A**:
| 特性 | SimpleUNet | 完整 UNet |
|------|-----------|-----------|
| 用途 | 测试扩散算法 | 实际训练和生成 |
| 复杂度 | 单层卷积 | 多层下采样+上采样 |
| 时间嵌入 | 简单扩展 | 正弦位置编码 + MLP |
| 注意力 | 无 | 多头自注意力 |
| 性能 | 很差 | 高质量生成 |

---

## 📁 相关文件列表

### 计划中的实现文件
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\models\__init__.py` - 模块初始化
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\models\unet.py` - U-Net 完整实现

### 相关文档
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\idea.md` - U-Net 规格说明（第 3.C 节）
- `D:\Tutorials\tutorials_for_claude_code\diffusion_model_implementations\coding_paradigm.md` - 编程规范

---

## 🔗 参考资源

### U-Net 原始论文
- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015

### 扩散模型中的 U-Net 变体
- [DDPM] Ho et al. - 基础 U-Net + 时间嵌入 + 注意力
- [Improved DDPM] Nichol & Dhariwal - 改进的架构和超参数
- [Guided Diffusion] Dhariwal & Nichol - 更深的网络和自适应归一化

### 实现参考
- PyTorch U-Net: https://github.com/milesial/Pytorch-UNet
- OpenAI Guided Diffusion: https://github.com/openai/guided-diffusion
- Hugging Face Diffusers: https://github.com/huggingface/diffusers

---

**下一步行动**: 实现完整的 UNet 类，包括时间嵌入、残差块、注意力模块和跳跃连接。
