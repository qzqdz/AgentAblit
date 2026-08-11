"""flashinfer 融合 NVFP4 线性层 + 模型改装工具 (GB10/sm_120+ 专用)。
- 离线把 nn.Linear.weight 量化为 NVFP4(per-tensor global scale + per-block fp8 scale)
- 前向: fp4_quantize(激活) 融合内核 + mm_fp4 (cutlass/cudnn) FP4 张量核 GEMM
- 实测真实 MLP 形状即使 M=1 decode 也 3-4x 于 bf16
用法:
    from fp4_linear import convert_to_fp4
    convert_to_fp4(model, targets=("gate_proj","up_proj","down_proj",
                                   "q_proj","k_proj","v_proj","o_proj"))
"""
import torch, torch.nn as nn
import flashinfer
from flashinfer import fp4_quantize, mm_fp4

FP4_MAX, FP8_MAX, BS = 6.0, 448.0, 16

def _calib_gs(t: torch.Tensor) -> torch.Tensor:
    amax = t.abs().amax().clamp(min=1e-8)
    return (FP8_MAX * FP4_MAX / amax).to(torch.float32).reshape(1)

class FP4Linear(nn.Module):
    """替换 nn.Linear: y = x @ W^T (NVFP4 融合内核)。权重离线量化, 推理期只量化激活。"""
    def __init__(self, lin: nn.Linear, backend: str = "cutlass"):
        super().__init__()
        W = lin.weight.data.to(torch.bfloat16).cuda()
        self.out_features, self.in_features = W.shape
        self.backend = backend
        w_gs = _calib_gs(W)
        w_fp4, w_sf = fp4_quantize(W, w_gs, sf_vec_size=BS)
        # 存为 buffer (不训练)
        self.register_buffer("w_fp4", w_fp4.contiguous(), persistent=False)
        self.register_buffer("w_sf",  w_sf.contiguous(),  persistent=False)
        self.register_buffer("w_gs",  w_gs,               persistent=False)
        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.data.to(torch.bfloat16).cuda(), persistent=False)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, self.in_features).to(torch.bfloat16).contiguous()
        x_gs = _calib_gs(x2)
        a, a_sf = fp4_quantize(x2, x_gs, sf_vec_size=BS)
        alpha = (1.0 / (x_gs * self.w_gs)).to(torch.float32)
        y = mm_fp4(a, self.w_fp4.T, a_sf, self.w_sf.T, alpha=alpha,
                   out_dtype=torch.bfloat16, block_size=BS, backend=self.backend)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*orig_shape[:-1], self.out_features)

def convert_to_fp4(model, targets, backend="cutlass", verbose=True):
    """把 model 中名字以 targets 任一结尾的 nn.Linear 替换为 FP4Linear。返回替换数。"""
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, FP4Linear(child, backend=backend))
                n += 1
    if verbose:
        print(f"[fp4] 替换 {n} 个 Linear 为 NVFP4 ({backend} 后端): {sorted(set(targets))}")
    return n
