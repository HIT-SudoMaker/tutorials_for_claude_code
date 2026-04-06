[根目录](../../../CLAUDE.md) > [diffusion_model_implementations](../CLAUDE.md) > **models**

# models - 神经网络模型模块

**最后更新时间**: 2026-04-06T20:36:06

---

## 变更日志 (Changelog)

### 2026-04-06
- 重新生成模块文档（原文件已删除后重建）
- 修正描述：当前不再包含 SimpleUNet，只有完整实现的 UNet 和 DiT

---

## 模块职责

本模块提供两种噪声预测网络架构，均支持 `class` 和 `none` 两种条件模式。所有模型共享统一的 `forward` 接口签名，输出形状与输入 `x_t` 一致。

---

## 入口和启动

- **`__init__.py`** -- 模块入口，提供 `build_model(model_config)` 工厂函数，根据 `config["type"]` 构建对应的模型实例
- 每个 `.py` 文件均可通过 `python unet.py` / `python dit.py` 独立运行内置形状验证测试

---

## 外部接口

### 统一 forward 签名

```python
def forward(self, x_t: torch.Tensor, t: torch.Tensor, condition: Optional[torch.Tensor]) -> torch.Tensor
```

### 工厂函数

```python
def build_model(model_config: Dict[str, Any]) -> nn.Module
```

校验规则：
- 必须包含 `type`、`in_channels`、`out_channels` 字段
- `in_channels` 必须与 `out_channels` 一致（保证输出形状与 x_t 一致）
- `condition.type` 仅支持 `"none"` 或 `"class"`
- `model.type` 仅支持 `"unet"` 或 `"dit"`

---

## 关键依赖和配置

### UNet (`unet.py`)

完整的编码器-瓶颈-解码器结构，包含以下组件：

| 类名 | 职责 |
|------|------|
| `SinusoidalTimeEmbedding` | 正弦时间步嵌入，将标量 t 映射为向量 |
| `_ResBlock` | 残差块，融合时间嵌入，含 GroupNorm + Conv + SiLU |
| `_AttentionBlock` | 自注意力块，用于瓶颈层捕获全局依赖 |
| `UNet` | 完整 UNet，含编码器、瓶颈、解码器、skip connection |

构造参数：`in_channels`, `out_channels`, `condition_type`, `num_classes`, `embedding_dim`, `base_channels` (64), `channel_multipliers` ((1,2,4)), `num_res_blocks` (2)

架构特点：
- 编码器和解码器各 3 个尺度（base_channels x 1/2/4）
- 每个尺度 2 个残差块
- 瓶颈层：ResBlock -> Attention -> ResBlock
- 最终拼接 init_conv 的 skip connection 后降维输出

### DiT (`dit.py`)

Diffusion Transformer 架构，使用 AdaLN（Adaptive Layer Normalization）进行时间与条件调制：

| 类名 | 职责 |
|------|------|
| `_PatchEmbedding` | 将图像切分为 patch 并投影为序列 |
| `_DiTBlock` | Transformer 块，含 AdaLN 调制的自注意力和前馈网络 |
| `_FinalLayer` | 最终输出层，将 patch 序列还原为图像 |
| `DiT` | 完整 DiT 模型 |

构造参数：`in_channels`, `out_channels`, `condition_type`, `num_classes`, `embedding_dim`, `patch_size` (2), `embed_dim` (256), `depth` (6), `num_heads` (8), `mlp_ratio` (4), `max_grid_size` (128)

架构特点：
- 1D 正弦位置编码通过外积求和构建 2D 位置编码
- _DiTBlock 的 AdaLN 生成 6 个调制参数（shift1, scale1, gate1, shift2, scale2, gate2）
- 输入图像的 H 和 W 必须能被 patch_size 整除

---

## 数据模型

### 条件模式

| 模式 | condition_type | class_embed | forward 时 condition 值 |
|------|---------------|-------------|------------------------|
| 无条件 | `"none"` | None | None |
| 类别条件 | `"class"` | `nn.Embedding(num_classes, dim)` | 类别索引张量 (B,) |

### 条件嵌入融合方式

- **UNet**: 时间嵌入 + 类别嵌入（直接相加），注入 ResBlock 的时间投影层
- **DiT**: 时间嵌入 + 类别嵌入（直接相加），通过 AdaLN 调制每个 Transformer 块和最终输出层

---

## 测试和质量

每个文件末尾包含 `if __name__ == "__main__"` 测试块，验证：

1. `class` 条件模式前向推理（输出形状与输入一致）
2. `none` 条件模式前向推理
3. 不同输入通道数（1 通道和 3 通道）
4. 不同输入分辨率（32x32 和 64x64）

`__init__.py` 末尾额外验证：
1. `build_model` 正常构建（class 和 none 模式）
2. 不支持的模型类型快速失败
3. 不支持的 condition.type 快速失败
4. 缺少必要字段快速失败
5. in_channels != out_channels 快速失败

---

## 常见问题 (FAQ)

**Q: UNet 是否还有 SimpleUNet？**
A: 没有。当前 `unet.py` 只包含完整的 `UNet` 类。如果需要更简单的架构，可以使用 `dit.py` 中的 `DiT` 并调小 `depth` 参数。

**Q: DiT 的输入分辨率有什么限制？**
A: 输入图像的 H 和 W 必须能被 `patch_size`（默认 2）整除。空间网格尺寸不能超过 `max_grid_size`（默认 128）。

**Q: 如何增加模型容量？**
A: UNet 可调大 `base_channels` 或 `channel_multipliers`；DiT 可调大 `embed_dim`、`depth` 或 `num_heads`。通过 `config.yaml` 的 `model.unet` 和 `model.dit` 字段传递。

---

## 相关文件列表

```
models/
├── __init__.py       # build_model 工厂函数（含参数校验和快速失败）
├── unet.py           # UNet（SinusoidalTimeEmbedding + _ResBlock + _AttentionBlock + UNet）
└── dit.py            # DiT（_PatchEmbedding + _DiTBlock + _FinalLayer + DiT）
```
