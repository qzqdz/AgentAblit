"""Make vLLM able to serve this Qwen3.5-9B text-only checkpoint.

Two gaps have to be bridged, and both must survive into vLLM's EngineCore *subprocess* —
which is why this lives in a real importable module and is registered by dotted string path
rather than by passing a class object. A class defined inside a function cannot be imported
by the child process; registration then silently no-ops there and vLLM falls back to the
in-tree multimodal `Qwen3_5ForConditionalGeneration`, which fails on a None multimodal_config.

  1. vLLM 0.23 *implements* `Qwen3_5ForCausalLM` in `vllm/model_executor/models/qwen3_5.py`
     but registers only the `*ForConditionalGeneration` variants for text generation.
  2. The checkpoint stores the text tower under `model.language_model.*` (Qwen3.5's base is
     VL-shaped) while vLLM's dense class expects `model.*`. Setting `hf_to_vllm_mapper` does
     NOT work: `Qwen3_5ForCausalLMBase.load_weights` builds its `AutoWeightsLoader` without
     passing a mapper, so the rename must happen on the weight iterator itself.

The checkpoint carries no mtp and no vision tensors (426 `model.language_model.*` plus
`lm_head.weight`), so a bare prefix swap is the entire conversion.
"""
from __future__ import annotations

import torch

from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)

_OLD_PREFIX = "model.language_model."
_NEW_PREFIX = "model."

ARCH_NAME = "Qwen3_5ForCausalLM"
CLASS_PATH = f"{__name__}:Qwen3_5TextForCausalLM"


class Qwen3_5TextForCausalLM(Qwen3_5ForCausalLM):
    """Dense Qwen3.5 causal LM: `model.language_model.*` weights + the hybrid-cache interface.

    The stock `Qwen3_5ForCausalLM` is missing `IsHybrid`, so vLLM never sizes a GatedDeltaNet
    state cache for it and `mamba/abstract.py` trips `assert mamba_block_size is not None`.
    24 of this model's 32 layers are GDN linear attention (`full_attention_interval=4`), so the
    state cache is not optional. The three members are lifted verbatim from the multimodal
    sibling, which does implement them; both read `model_config.hf_text_config`, which for a
    text-only checkpoint is the config itself, so they transfer unchanged.
    """

    is_hybrid = True
    get_mamba_state_shape_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config.__func__
    )
    get_mamba_state_dtype_from_config = classmethod(
        Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config.__func__
    )
    get_mamba_state_copy_func = Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func

    # The config keeps the VL base's M-RoPE block (mrope_section [11,11,10],
    # mrope_interleaved, partial_rotary_factor 0.25), so gpu_model_runner asserts
    # `supports_mrope(model)`. The rope layout is load-bearing and must not be disabled --
    # but with no multimodal features the sibling's implementation reduces exactly to
    # arange(n) broadcast over the 3 sections with a zero delta (its loop body never runs and
    # the trailing text branch supplies the whole sequence). Borrowing the real method is not
    # an option: it eagerly reads config.vision_config.spatial_merge_size, which a text-only
    # checkpoint does not have.
    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        n = len(input_tokens)
        positions = torch.arange(n, dtype=torch.int64).unsqueeze(0).expand(3, -1)
        return positions.contiguous(), 0

    def load_weights(self, weights):
        def _renamed():
            for name, tensor in weights:
                if name.startswith(_OLD_PREFIX):
                    name = _NEW_PREFIX + name[len(_OLD_PREFIX):]
                yield name, tensor

        return super().load_weights(_renamed())


def register() -> str:
    """Register the remapped arch with vLLM. Safe to call more than once."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(ARCH_NAME, CLASS_PATH)
    return f"{ARCH_NAME} -> {CLASS_PATH}"
