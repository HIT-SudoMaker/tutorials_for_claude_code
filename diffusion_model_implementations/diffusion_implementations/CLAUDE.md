[根目录](../../../CLAUDE.md) > [diffusion_model_implementations](../CLAUDE.md) > **diffusion_implementations**

# diffusion_implementations - 扩散算法模块

**最后更新时间**: 2026-04-06T20:36:06

---

## 变更日志 (Changelog)

### 2026-04-06
- 重新生成模块文档（原文件已删除后重建）

---

## 模块职责

本模块提供四种扩散采样算法的统一接口实现。所有算法均继承 `torch.nn.Module`，共享相同的方法签名，支持通过配置文件无缝切换。

---

## 入口和启动

- **`__init__.py`** -- 模块入口，提供 `build_diffusion(config)` 工厂函数，根据 `config["implementation"]` 构建对应的算法实例
- 每个 `.py` 文件均可通过 `python ddpm.py` 等方式独立运行内置形状验证测试

---

## 外部接口

### 统一方法签名

```python
def forward_sample(self, x_0: torch.Tensor, t: int) -> torch.Tensor
def reverse_sample(self, x_t: torch.Tensor, t: int, condition: Optional[torch.Tensor], model: nn.Module) -> torch.Tensor
def reverse_sample_loop(self, shape: Tuple[int, ...], condition: Optional[torch.Tensor], model: nn.Module) -> torch.Tensor
```

### 工厂函数

```python
def build_diffusion(config: Dict[str, Any]) -> nn.Module
```

校验规则：
- 必须包含 `implementation`、`timesteps`、`beta_start`、`beta_end` 字段
- `timesteps` 必须为正整数
- `beta_start` 和 `beta_end` 必须为 (0, 1) 之间的数值，且 `beta_start < beta_end`
- `dpm_solver.order` 仅支持 1 或 2

---

## 关键依赖和配置

- **DDPM** (`ddpm.py`) -- 基础算法，其他三种算法均依赖其预计算的噪声调度系数
- **DDIM** (`ddim.py`) -- 依赖 DDPM 的 alphas_cumprod 系数，额外参数 `eta`（随机性控制）、`steps`（子序列步数）
- **SDE Solver** (`sde_solver.py`) -- 依赖 DDPM 的后验分布系数，使用 Euler-Maruyama 离散化
- **DPM-Solver** (`dpm_solver.py`) -- 依赖 DDPM 的 alphas_cumprod，额外参数 `order`（1/2）、`steps`（子序列步数）

所有算法共享以下构造参数：
- `timesteps` (int) -- 总扩散步数 T
- `beta_start` (float) -- 噪声调度起始值
- `beta_end` (float) -- 噪声调度终止值
- `beta_schedule` (str) -- 调度策略，`"linear"` 或 `"cosine"`

---

## 数据模型

### DDPM 内部预计算系数

| 缓冲区 | 含义 |
|--------|------|
| `betas` | beta 序列 |
| `alphas` | alpha 序列 (1 - beta) |
| `alphas_cumprod` | alpha 累积乘积 |
| `sqrt_alphas_cumprod` | 前向加噪系数 |
| `sqrt_one_minus_alphas_cumprod` | 前向加噪噪声系数 |
| `posterior_variance` | 反向过程后验方差 |
| `posterior_mean_coef1` / `coef2` | 反向过程后验均值系数 |

### DDPM -> DDIM/SDESolver/DPMSolver 复用关系

DDIM、SDESolver、DPMSolver 均通过 `DDPM(timesteps, beta_start, beta_end, beta_schedule)` 实例化一个内部 DDPM 对象，然后从其 buffer 中提取所需的噪声调度系数。这种设计避免了系数计算逻辑的重复。

---

## 测试和质量

每个文件末尾包含 `if __name__ == "__main__"` 测试块，使用 `_DummyModel`（返回随机张量的占位模型）验证：

1. `forward_sample` 输入输出形状一致
2. `reverse_sample` 输入输出形状一致
3. `reverse_sample_loop` 生成指定形状的样本
4. `cosine` 调度策略正常工作
5. 无条件生成（condition=None）正常工作

DDIM 额外验证 `eta=1.0` 随机性采样。DPMSolver 额外验证 `order=1` 一阶求解。

---

## 相关文件列表

```
diffusion_implementations/
├── __init__.py       # build_diffusion 工厂函数
├── ddpm.py           # DDPM 算法（基础算法，含噪声调度系数预计算）
├── ddim.py           # DDIM 算法（确定性/随机性子序列采样器）
├── sde_solver.py     # SDE Solver（Euler-Maruyama 离散化）
└── dpm_solver.py     # DPM-Solver（多步加速采样器）
```
