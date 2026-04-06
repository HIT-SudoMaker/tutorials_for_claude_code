import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    将标量时间步映射为向量表示的正弦嵌入层
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        将时间步编码为向量

        Args:
            t: 形状为 (B,) 的整数时间步张量

        Returns:
            形状为 (B, embed_dim*4) 的嵌入向量
        """
        half_dim = self.embed_dim // 2
        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -embedding)
        embedding = t.float().unsqueeze(-1) * embedding.unsqueeze(0)
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        return self.mlp(embedding)


class _ResBlock(nn.Module):
    """
    残差块，融合时间嵌入和可选的条件嵌入
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """
        融合时间嵌入并输出残差结果

        Args:
            x:              形状为 (B, C_in, H, W) 的特征图
            time_embedding: 形状为 (B, time_embed_dim) 的时间嵌入

        Returns:
            形状为 (B, C_out, H, W) 的输出特征图
        """
        hidden = self.act(self.norm1(x))
        hidden = self.conv1(hidden)
        hidden = hidden + self.time_proj(self.act(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        hidden = self.act(self.norm2(hidden))
        hidden = self.conv2(hidden)
        return hidden + self.residual_conv(x)


class _AttentionBlock(nn.Module):
    """
    自注意力块，用于UNet中间层捕获全局依赖
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行自注意力并返回残差连接结果

        Args:
            x: 形状为 (B, C, H, W) 的特征图

        Returns:
            形状为 (B, C, H, W) 的输出特征图
        """
        B, C, H, W = x.shape
        hidden = self.norm(x)
        hidden = hidden.flatten(2).transpose(1, 2)
        hidden, _ = self.attention(hidden, hidden, hidden)
        hidden = hidden.transpose(1, 2).reshape(B, C, H, W)
        return x + hidden


class UNet(nn.Module):
    """
    完整UNet噪声预测网络，包含下采样、上采样和skip connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_type: str = "none",
        num_classes: int = 10,
        embedding_dim: int = 128,
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
    ) -> None:
        """
        构造UNet并初始化编码器-瓶颈-解码器结构

        Args:
            in_channels:         输入通道数
            out_channels:        输出通道数
            condition_type:      条件类型，可选"none"或"class"
            num_classes:         类别条件下的类别数
            embedding_dim:       时间步嵌入的基础维度
            base_channels:       基础通道数
            channel_multipliers: 各层通道数倍率
            num_res_blocks:      每个尺度的残差块数量
        """
        super().__init__()
        self.condition_type = condition_type
        self.time_embed = SinusoidalTimeEmbedding(base_channels)

        if condition_type == "class":
            self.class_embed = nn.Embedding(num_classes, base_channels * 4)
        else:
            self.class_embed = None

        # 构建多尺度通道数列表
        channels = [base_channels * m for m in channel_multipliers]

        # 编码器各层
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        # 初始卷积
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        prev_channels = base_channels
        for i, channel in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(_ResBlock(prev_channels, channel, base_channels * 4))
                prev_channels = channel
            self.encoder_blocks.append(blocks)
            if i < len(channels) - 1:
                self.downsamplers.append(nn.Conv2d(channel, channel, 3, stride=2, padding=1))

        # 瓶颈
        self.bottleneck = nn.ModuleList([
            _ResBlock(prev_channels, prev_channels, base_channels * 4),
            _AttentionBlock(prev_channels),
            _ResBlock(prev_channels, prev_channels, base_channels * 4),
        ])

        # 解码器各层
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for i, channel in enumerate(reversed(channels)):
            blocks = nn.ModuleList()
            # 首个解码块多接收skip connection的通道
            skip_channels = channel + prev_channels
            blocks.append(_ResBlock(skip_channels, channel, base_channels * 4))
            for _ in range(num_res_blocks - 1):
                blocks.append(_ResBlock(channel, channel, base_channels * 4))
            self.decoder_blocks.append(blocks)
            prev_channels = channel
            if i < len(channels) - 1:
                self.upsamplers.append(
                    nn.ConvTranspose2d(channel, channel, 4, stride=2, padding=1)
                )

        # 最终拼接层：将解码器最后一层输出与init_conv的skip拼接后降维
        self.final_skip_conv = _ResBlock(
            base_channels + base_channels, base_channels, base_channels * 4
        )
        # 最终输出卷积
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        预测噪声，输出形状与x_t一致

        Args:
            x_t:       形状为 (B, C_in, H, W) 的带噪数据
            t:         形状为 (B,) 的整数时间步
            condition: 形状为 (B,) 的类别索引张量，当condition_type为"none"时为None

        Returns:
            形状为 (B, C_out, H, W) 的预测噪声
        """
        time_embedding = self.time_embed(t)

        if self.class_embed is not None and condition is not None:
            time_embedding = time_embedding + self.class_embed(condition)

        # 编码器，逐层保存skip connection
        skips: List[torch.Tensor] = []
        hidden = self.init_conv(x_t)
        skips.append(hidden)

        for i, blocks in enumerate(self.encoder_blocks):
            for block in blocks:
                hidden = block(hidden, time_embedding)
            skips.append(hidden)
            if i < len(self.downsamplers):
                hidden = self.downsamplers[i](hidden)

        # 瓶颈
        hidden = self.bottleneck[0](hidden, time_embedding)
        hidden = self.bottleneck[1](hidden)
        hidden = self.bottleneck[2](hidden, time_embedding)

        # 解码器，逐层拼接skip connection
        for i, blocks in enumerate(self.decoder_blocks):
            hidden = torch.cat([hidden, skips[-(i + 1)]], dim=1)
            for block in blocks:
                hidden = block(hidden, time_embedding)
            if i < len(self.upsamplers):
                hidden = self.upsamplers[i](hidden)

        # 拼接初始skip connection并降维
        hidden = torch.cat([hidden, skips[0]], dim=1)
        hidden = self.final_skip_conv(hidden, time_embedding)
        hidden = self.final_conv(hidden)

        return hidden


if __name__ == "__main__":
    print("测试 UNet (condition_type='class') ...")
    model_c = UNet(
        in_channels=1, out_channels=1,
        condition_type="class", num_classes=10, embedding_dim=128,
    )
    x = torch.randn(2, 1, 32, 32)
    t = torch.randint(0, 1000, (2,))
    c = torch.randint(0, 10, (2,))
    out_c = model_c(x, t, c)
    assert out_c.shape == x.shape, f"UNet class模式输出形状不匹配: {out_c.shape} vs {x.shape}"
    print(f"  输入: {x.shape}, 输出: {out_c.shape} -- PASS")

    print("测试 UNet (condition_type='none') ...")
    model_d = UNet(
        in_channels=3, out_channels=3,
        condition_type="none",
    )
    x3 = torch.randn(2, 3, 64, 64)
    out_d = model_d(x3, t, None)
    assert out_d.shape == x3.shape, f"UNet none模式输出形状不匹配: {out_d.shape} vs {x3.shape}"
    print(f"  输入: {x3.shape}, 输出: {out_d.shape} -- PASS")

    print("\n所有unet.py测试通过")
