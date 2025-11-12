"""
U-Net 模型实现模块。

本模块提供两种 U-Net 架构：
1. SimpleUNet: 简化版，用于快速测试扩散算法
2. UNet: 完整版，包含时间嵌入、注意力机制、残差连接
"""

from typing import List, Optional
import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    """
    简化版 U-Net 模型，用于测试扩散算法的噪声预测功能。

    该模型通过简单的卷积层模拟 U-Net 的行为，将带噪数据、时间步和条件信息
    融合后预测噪声。主要用于验证扩散算法的接口正确性，而非追求生成质量。

    Args:
        in_channels (int): 输入数据的通道数（灰度图为 1，RGB 图为 3）。
        out_channels (int): 输出噪声的通道数（通常与输入通道数相同）。

    Example:
        >>> model = SimpleUNet(in_channels=1, out_channels=1)
        >>> x_t = torch.randn(4, 1, 28, 28)  # batch_size=4, MNIST尺寸
        >>> t = torch.randint(0, 1000, (4,))
        >>> condition = torch.randn(4, 1, 28, 28)
        >>> noise_pred = model(x_t, t, condition)
        >>> print(noise_pred.shape)  # torch.Size([4, 1, 28, 28])
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 简单卷积层：输入包括 x_t (in_channels) + t (1) + condition (in_channels)
        # 因此总输入通道数为 in_channels * 2 + 1
        self.conv = nn.Conv2d(
            in_channels * 2 + 1,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        预测给定带噪数据在当前时间步的噪声。

        Args:
            x_t (torch.Tensor): 当前时间步的带噪数据。
                shape: (batch_size, in_channels, height, width)
            t (torch.Tensor): 当前时间步索引。
                shape: (batch_size,)
            condition (torch.Tensor): 条件信息（用于条件生成）。
                shape: (batch_size, in_channels, height, width)
                注意：对于类别条件，需要先通过嵌入层转换为张量

        Returns:
            torch.Tensor: 预测的噪声，与 x_t 形状完全相同。
                shape: (batch_size, out_channels, height, width)
        """
        # 将时间步 t 嵌入为与 x_t 空间维度相同的张量
        # t: (batch_size,) -> (batch_size, 1, 1, 1) -> (batch_size, 1, H, W)
        t_embedding = t.view(t.size(0), 1, 1, 1).expand(
            -1, 1, x_t.size(2), x_t.size(3)
        )

        # 拼接输入：[x_t, t_embedding, condition]
        # shape: (batch_size, in_channels*2 + 1, height, width)
        concatenated_input = torch.cat([x_t, t_embedding, condition], dim=1)

        # 通过卷积层预测噪声
        noise_prediction = self.conv(concatenated_input)

        return noise_prediction


class UNet(nn.Module):
    """
    完整的 U-Net 模型，包含时间嵌入、注意力机制、残差连接。

    该模型是生产级的噪声预测网络，包含以下组件：
    - 正弦位置编码的时间步嵌入
    - 类别条件嵌入（支持条件生成）
    - 多层下采样和上采样路径
    - 残差块（ResNet Block）
    - 自注意力模块（在指定分辨率层）
    - 跳跃连接（Skip Connections）

    Args:
        in_channels (int): 输入数据的通道数。
        out_channels (int): 输出噪声的通道数。
        base_channels (int, optional): 基础通道数。默认为 64。
        channel_multipliers (List[int], optional): 每层的通道数倍增系数。
            默认为 [1, 2, 4, 8]（4层下采样）。
        attention_resolutions (List[int], optional): 在哪些分辨率层添加自注意力。
            默认为 [16, 8]。
        num_residual_blocks (int, optional): 每层的残差块数量。默认为 2。
        dropout (float, optional): Dropout 概率。默认为 0.1。
        num_classes (Optional[int], optional): 类别数量（用于类别条件生成）。
            如果为 None，则不使用类别条件。默认为 None。

    Example:
        >>> # 无条件生成
        >>> model = UNet(in_channels=1, out_channels=1)
        >>> x_t = torch.randn(4, 1, 28, 28)
        >>> t = torch.randint(0, 1000, (4,))
        >>> condition = torch.zeros(4, 1, 28, 28)  # 无条件时传入零张量
        >>> noise_pred = model(x_t, t, condition)

        >>> # 类别条件生成
        >>> model = UNet(in_channels=1, out_channels=1, num_classes=10)
        >>> class_labels = torch.randint(0, 10, (4,))  # MNIST 数字类别
        >>> # 需要先通过嵌入层处理 class_labels 得到 condition
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 64,
        channel_multipliers: List[int] = None,
        attention_resolutions: List[int] = None,
        num_residual_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: Optional[int] = None
    ) -> None:
        super().__init__()

        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]
        if attention_resolutions is None:
            attention_resolutions = [16, 8]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_classes = num_classes

        # TODO: 实现完整的 U-Net 架构
        # 当前为占位符实现，将在任务 3.1 中完成
        # 主要组件包括：
        # 1. 时间步嵌入层 (TimeEmbedding)
        # 2. 类别条件嵌入层（可选）
        # 3. 下采样路径（Downsample + ResidualBlock + AttentionBlock）
        # 4. 瓶颈层（Bottleneck）
        # 5. 上采样路径（Upsample + ResidualBlock + AttentionBlock + Skip）
        # 6. 输出层

        # 临时占位符：简单卷积层
        self.placeholder_conv = nn.Conv2d(
            in_channels * 2 + 1,
            out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        预测给定带噪数据在当前时间步的噪声。

        Args:
            x_t (torch.Tensor): 当前时间步的带噪数据。
                shape: (batch_size, in_channels, height, width)
            t (torch.Tensor): 当前时间步索引。
                shape: (batch_size,)
            condition (torch.Tensor): 条件信息。
                - 类别条件：shape (batch_size,)，需要先通过嵌入层
                - 张量条件：shape (batch_size, in_channels, height, width)

        Returns:
            torch.Tensor: 预测的噪声，与 x_t 形状完全相同。
                shape: (batch_size, out_channels, height, width)
        """
        # TODO: 实现完整的前向传播逻辑（任务 3.1）
        # 当前为临时实现，与 SimpleUNet 相同

        # 时间步嵌入
        t_embedding = t.view(t.size(0), 1, 1, 1).expand(
            -1, 1, x_t.size(2), x_t.size(3)
        )

        # 拼接输入
        concatenated_input = torch.cat([x_t, t_embedding, condition], dim=1)

        # 临时使用简单卷积
        noise_prediction = self.placeholder_conv(concatenated_input)

        return noise_prediction


# ============================================================================
# 测试模块
# ============================================================================

if __name__ == "__main__":
    # 设置 UTF-8 编码以支持中文输出
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("SimpleUNet 和 UNet 模型测试")
    print("=" * 70)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}\n")

    # 测试参数（MNIST 尺寸）
    batch_size = 4
    in_channels = 1
    out_channels = 1
    height = 28
    width = 28
    num_timesteps = 1000

    # 创建测试数据
    x_t = torch.randn(batch_size, in_channels, height, width).to(device)
    t = torch.randint(0, num_timesteps, (batch_size,)).to(device)
    condition = torch.randn(batch_size, in_channels, height, width).to(device)

    print("-" * 70)
    print("1. 测试 SimpleUNet")
    print("-" * 70)

    # 实例化 SimpleUNet
    simple_model = SimpleUNet(
        in_channels=in_channels,
        out_channels=out_channels
    ).to(device)

    # 前向传播
    with torch.no_grad():
        noise_pred = simple_model(x_t, t, condition)

    print(f"输入 x_t 形状: {x_t.shape}")
    print(f"时间步 t 形状: {t.shape}")
    print(f"条件 condition 形状: {condition.shape}")
    print(f"输出 noise_pred 形状: {noise_pred.shape}")
    print(f"✅ SimpleUNet 测试通过！\n")

    print("-" * 70)
    print("2. 测试 UNet (临时占位符实现)")
    print("-" * 70)

    # 实例化 UNet
    unet_model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=64,
        channel_multipliers=[1, 2, 4, 8],
        attention_resolutions=[16, 8],
        num_residual_blocks=2,
        dropout=0.1,
        num_classes=10  # MNIST 类别数
    ).to(device)

    # 前向传播
    with torch.no_grad():
        noise_pred_unet = unet_model(x_t, t, condition)

    print(f"输入 x_t 形状: {x_t.shape}")
    print(f"时间步 t 形状: {t.shape}")
    print(f"条件 condition 形状: {condition.shape}")
    print(f"输出 noise_pred 形状: {noise_pred_unet.shape}")
    print(f"⚠️  UNet 当前为占位符实现，将在任务 3.1 中完成\n")

    print("=" * 70)
    print("模型测试完成！")
    print("=" * 70)
