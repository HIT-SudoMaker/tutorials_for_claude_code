import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class _PatchEmbedding(nn.Module):
    """
    将二维图像切分为patch并投影为向量序列
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        将图像切分为patch并投影为序列

        Args:
            x: 形状为 (B, C, H, W) 的图像张量

        Returns:
            seq: 形状为 (B, N, D) 的patch序列，N=H*W/patch_size^2
            spatial_size: (grid_h, grid_w) 空间网格尺寸
        """
        B, C, H, W = x.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        seq = self.proj(x).flatten(2).transpose(1, 2)
        return seq, (grid_h, grid_w)


class _DiTBlock(nn.Module):
    """
    DiT Transformer块，包含自注意力和前馈网络，均受时间与条件调制
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )
        # 可学习的调制参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim),
        )

    def forward(self, x: torch.Tensor, condition_embedding: torch.Tensor) -> torch.Tensor:
        """
        执行AdaLN调制后的自注意力与前馈计算

        Args:
            x:                   形状为 (B, N, D) 的序列
            condition_embedding: 形状为 (B, D) 的条件嵌入向量

        Returns:
            形状为 (B, N, D) 的输出序列
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(
            condition_embedding
        ).chunk(6, dim=-1)

        # 自注意力分支
        hidden = self.norm1(x)
        hidden = hidden * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        hidden, _ = self.attention(hidden, hidden, hidden)
        x = x + gate1.unsqueeze(1) * hidden

        # 前馈网络分支
        hidden = self.norm2(x)
        hidden = hidden * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        hidden = self.mlp(hidden)
        x = x + gate2.unsqueeze(1) * hidden

        return x


class _FinalLayer(nn.Module):
    """
    DiT最终输出层，将patch序列还原为图像
    """

    def __init__(self, embed_dim: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, patch_size * patch_size * out_channels)
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 2 * embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition_embedding: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """
        将patch序列还原为图像张量

        Args:
            x:                   形状为 (B, N, D) 的序列
            condition_embedding: 形状为 (B, D) 的条件嵌入
            grid_h:              空间网格高度
            grid_w:              空间网格宽度

        Returns:
            形状为 (B, out_channels, grid_h*patch_size, grid_w*patch_size) 的图像
        """
        shift, scale = self.adaLN_modulation(condition_embedding).chunk(2, dim=-1)
        hidden = self.norm_final(x)
        hidden = hidden * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        hidden = self.linear(hidden)
        B, N, _ = hidden.shape
        patch_size = self.patch_size
        hidden = hidden.reshape(B, grid_h, grid_w, patch_size, patch_size, self.out_channels)
        hidden = hidden.permute(0, 5, 1, 3, 2, 4).reshape(
            B, self.out_channels, grid_h * patch_size, grid_w * patch_size
        )
        return hidden


class DiT(nn.Module):
    """Diffusion Transformer噪声预测网络"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_type: str = "none",
        num_classes: int = 10,
        embedding_dim: int = 128,
        patch_size: int = 2,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        max_grid_size: int = 128,
    ) -> None:
        """
        初始化DiT模型

        Args:
            in_channels:    输入图像通道数
            out_channels:   输出噪声通道数
            condition_type: 条件类型，可选"none"或"class"
            num_classes:    类别条件下的类别数
            embedding_dim:  时间步嵌入的基础维度
            patch_size:     patch边长
            embed_dim:      Transformer嵌入维度
            depth:          Transformer块数量
            num_heads:      多头注意力的头数
            mlp_ratio:      前馈网络扩展倍率
            max_grid_size:  位置编码支持的最大空间网格尺寸
        """
        super().__init__()
        self.condition_type = condition_type
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 时间步嵌入
        half_dim = embedding_dim // 2
        frequency = math.log(10000) / (half_dim - 1)
        self.register_buffer(
            "time_freq",
            torch.exp(torch.arange(half_dim) * -frequency).float(),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # 条件嵌入
        if condition_type == "class":
            self.class_embed = nn.Embedding(num_classes, embed_dim)
        else:
            self.class_embed = None

        # patch嵌入
        self.patch_embed = _PatchEmbedding(in_channels, patch_size, embed_dim)

        # Transformer主体
        self.blocks = nn.ModuleList([
            _DiTBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 最终输出层
        self.final_layer = _FinalLayer(embed_dim, patch_size, out_channels)

        # 预计算patch网格的1D位置编码
        self._max_grid_size = max_grid_size
        pos = torch.arange(self._max_grid_size).float().unsqueeze(1)
        freq_dim = torch.arange(embed_dim // 2).float()
        pos_encoding = torch.zeros(self._max_grid_size, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(pos * 10000 ** (-freq_dim / (embed_dim // 2)))
        pos_encoding[:, 1::2] = torch.cos(pos * 10000 ** (-freq_dim / (embed_dim // 2)))
        self.register_buffer("pos_embed_1d", pos_encoding)

    def _build_pos_embed(self, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
        """
        构建2D位置编码

        Args:
            grid_h: 空间网格高度
            grid_w: 空间网格宽度
            device: 目标设备

        Returns:
            形状为 (grid_h * grid_w, embed_dim) 的位置编码
        """
        pe_h = self.pos_embed_1d[:grid_h]
        pe_w = self.pos_embed_1d[:grid_w]
        # 外积求和构成2D位置编码
        pos = pe_h.unsqueeze(1) + pe_w.unsqueeze(0)
        pos = pos.reshape(grid_h * grid_w, self.embed_dim)
        return pos.to(device)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        预测噪声，输出形状与x_t一致

        Args:
            x_t:       形状为 (B, C_in, H, W) 的带噪数据，H和W必须能被patch_size整除
            t:         形状为 (B,) 的整数时间步
            condition: 形状为 (B,) 的类别索引张量，当condition_type为"none"时为None

        Returns:
            形状为 (B, C_out, H, W) 的预测噪声
        """
        B, C, H, W = x_t.shape

        # 编码时间步
        time_embedding = t.float().unsqueeze(-1) * self.time_freq.unsqueeze(0)
        time_embedding = torch.cat([torch.sin(time_embedding), torch.cos(time_embedding)], dim=-1)
        condition_embedding = self.time_mlp(time_embedding)

        # 融合条件嵌入
        if self.class_embed is not None and condition is not None:
            condition_embedding = condition_embedding + self.class_embed(condition)

        # patch嵌入与位置编码
        seq, (grid_h, grid_w) = self.patch_embed(x_t)
        if grid_h > self._max_grid_size or grid_w > self._max_grid_size:
            raise ValueError(
                f"空间网格尺寸({grid_h}, {grid_w})超过max_grid_size={self._max_grid_size}，"
                f"请增大max_grid_size或减小输入分辨率"
            )
        pos = self._build_pos_embed(grid_h, grid_w, seq.device)
        seq = seq + pos.unsqueeze(0)

        # Transformer主体
        for block in self.blocks:
            seq = block(seq, condition_embedding)

        # 还原为图像
        out = self.final_layer(seq, condition_embedding, grid_h, grid_w)

        return out


if __name__ == "__main__":
    # DiT测试 (condition_type='class')
    print("测试 DiT (condition_type='class') ...")
    model_a = DiT(
        in_channels=1, out_channels=1,
        condition_type="class", num_classes=10, embedding_dim=128,
        patch_size=2, embed_dim=256, depth=4, num_heads=8,
    )
    x = torch.randn(2, 1, 32, 32)
    t = torch.randint(0, 1000, (2,))
    c = torch.randint(0, 10, (2,))
    out = model_a(x, t, c)
    assert out.shape == x.shape, f"DiT class模式输出形状不匹配: {out.shape} vs {x.shape}"
    print(f"  输入: {x.shape}, 输出: {out.shape} -- PASS")

    # DiT测试 (condition_type='none')
    print("测试 DiT (condition_type='none') ...")
    model_b = DiT(
        in_channels=3, out_channels=3,
        condition_type="none",
        patch_size=2, embed_dim=128, depth=4, num_heads=4,
    )
    x3 = torch.randn(2, 3, 64, 64)
    out3 = model_b(x3, t, None)
    assert out3.shape == x3.shape, f"DiT none模式输出形状不匹配: {out3.shape} vs {x3.shape}"
    print(f"  输入: {x3.shape}, 输出: {out3.shape} -- PASS")

    print("\n所有dit.py测试通过")
