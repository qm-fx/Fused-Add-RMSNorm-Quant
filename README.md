# Fused-Add-RMSNorm-Quant

基于 CUDA 手写的融合算子，在单次 kernel launch 中完成 LLM 推理中Transformer 层的 **残差相加 + RMSNorm + FP8(E4M3) 量化**，针对 Hidden Size 较小（N = 128~384）的推理负载做了并行策略定制。

## 为什么需要融合

未融合时，这段处理需要三个独立的算子串联执行：

| 算子 | 显存读 | 显存写 | HBM 往返次数 |
|:---:|:---:|:---:|:---:|
| Residual Add | x, residual | tmp | 3 |
| RMSNorm | tmp | out | 2 |
| Quant | out | qout | 2 |

三个算子均为访存受限，合计约 7 次全量张量的 HBM 往返；
融合后仅需读 `x`、`residual`，写 `qout`、更新后的 `residual`，流量降至约 4 次，同时消除 2 次 kernel launch 与中间张量的分配/读写开销。

## 性能表现

测试环境：NVIDIA A800 80GB，HBM 理论峰值带宽约 2039 GB/s。

| 用例 (M × N) | Baseline 耗时 | 融合耗时 | 加速比 | 等效带宽 | 占峰值 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 458752 × 128 | 3.076 ms | 0.359 ms | 8.56× | ~1635.66 GB/s | 80.22% |
| 294912 × 256 | 3.833 ms | 0.468 ms | 8.19× | ~1613.19 GB/s | 79.12% |
| 196608 × 384 | 3.855 ms | 0.546 ms | 7.06× | ~1382.74 GB/s | 67.81% |

> Baseline 为 PyTorch eager 下 Add → RMSNorm → Quant 三算子串联调用。

## 核心设计

### 并行策略：Warp-per-Row

每个 Warp 独立处理一行（一个 token 的 hidden 向量），每个 Block覆盖 16 行。选择该策略的动机是针对小 N 场景的针对性优化：

- **N 较小时行内数据量有限**，一个 Warp（32 线程）正好能覆盖，
  每行由同 Warp 的线程协作，不需要跨 Warp 通信；
- **行间完全独立**，行与行之间天然并行，Warp 之间无需任何同步；
- **硬件资源匹配**：一个 Block 只需 16 个 Warp，occupancy 和
  寄存器压力都容易控制在合理范围。

### 行内计算：单次加载、寄存器常驻

行内数据一次性从 HBM 读入寄存器后，在整条计算链中不再回写显存：

1. **Residual Add**：`x + residual`，结果保留在寄存器中；
2. **RMSNorm**：需要行内平方和进行归约；
3. **Quant**：归约得到 RMS 后，对寄存器中的数据做 FP8(E4M3) 量化。

### 归约策略：Warp 级蝶形归约

RMSNorm 需要行内平方和，一行的所有数据都在同一warp的线程内，通过 **Warp 级蝶形归约**：利用 `__shfl_xor_sync` 在Warp 内的 32 个线程之间做数据交换。无需 shared memory 和 `__syncthreads()`，避免了块内同步开销，是小 N 场景下更优的选择。

### FP8(E4M3) 量化

计算 scale 并做饱和处理（截断到 E4M3 可表示范围），输出 **截断后的float** 视具体实现而定（按实际填写），精度策略与参考实现误差满足校验阈值（1*10^-6）。