#!/usr/bin/env python3
"""Container-start patches for Mistral-Large-3 on vllm-torchtpu v0.32.0 (applied on every host before vLLM starts).

Default: PATCH 1 only. All benchmark results in this repository were produced with PATCH 1 only.
PATCHES 2-5 (Eagle speculative-decoding draft model) are opt-in: set ML3_EAGLE_PATCHES=1.

PATCH 1 (FP8 MoE weight-scale names, compressed-tensors):
  Mistral-Large-3-675B-Instruct-2512 (FP8) is a compressed-tensors `float-quantized` checkpoint with 128x128 block
  scales stored as `<proj>.weight_scale`. vllm_torchtpu dispatches compressed-tensors FP8 MoE layers to
  VllmFp8MoEMethodTPU (the `fp8` quant-method class), whose upstream Fp8MoEMethod.create_weights() registers block
  scales under the DeepSeek naming `w13_weight_scale_inv` / `w2_weight_scale_inv`. The expert params mapping then
  produces `...experts.routed_experts.w13_weight_scale`, which does not exist ->
  KeyError in deepseek_v2.py load_weights. Upstream vLLM's GPU compressed-tensors FP8 MoE method uses
  `w13_weight_scale` for block scales, so we set `weight_scale_name = "weight_scale"` on the TPU method instance built
  by the compressed-tensors dispatcher. Everything downstream already accepts either name:
  `_process_fp8_moe_weights` reads `w13_weight_scale_inv` or falls back to `w13_weight_scale`, deletes the checkpoint
  scale param via `w13_{weight_scale_name}`, and installs the runtime per-channel scale as `w13_weight_scale_inv`
  (which the GMM / fused-EP kernels read). No numerics change: same dequant(block FP8) -> requant(per-channel FP8).

PATCHES 2-5 (opt-in, ML3_EAGLE_PATCHES=1; not used for any result in this repository):
  Adapt the vLLM 0.32.0 Mistral-Large-3 Eagle draft model (mistral_large_3_eagle.py) and the vllm-torchtpu MLA op to
  the Mistral-Large-3-675B-Instruct-2512-Eagle draft (1 MoE layer, per-tensor FP8 weights, static activation scales).
  Eagle speculative decoding does not run end-to-end on this stack yet.
  2. EagleMistralLarge3Model: set `is_fused_shared_expert_enabled` (read by the inherited DeepseekV2Model.load_weights;
     the class calls nn.Module.__init__ instead of DeepseekV2Model.__init__) and the single-hidden-state attributes
     `use_aux_hidden_state` / `fc_input_size` read by vllm-torchtpu's Eagle proposer.
  3. EagleMistralLarge3ForCausalLM: build the draft quantization config with vllm-torchtpu's
     get_tpu_quantization_config() instead of the upstream CUDA Fp8Config.
  4. EagleMistralLarge3ForCausalLM: add an identity combine_hidden_states().
  5. mla_attention_op.py: while kv_b_proj is temporarily un-flipped for MLA weight processing, treat a per-tensor FP8
     quant method (block_quant=False) as block-quantized so the dequantized weight is not transposed a second time.
     Layers whose kv_b_proj is block-quantized or unquantized are unaffected.
"""
from __future__ import annotations

import os
import pathlib
import py_compile
import sys

ROOT = pathlib.Path("/workspace/vllm-torchtpu/src/vllm_torchtpu")
MARK = "[ml3-patch]"

PATCHES = [
    (
        ROOT / "layers/adapter/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py",
        """        if _is_weight_fp8(weight_quant):
            fp8_config = _build_fp8_config(weight_quant, input_quant)
            return VllmFp8MoEMethodTPU(fp8_config, fp8_config.get_moe_config(layer))
""",
        """        if _is_weight_fp8(weight_quant):
            fp8_config = _build_fp8_config(weight_quant, input_quant)
            _ml3_method = VllmFp8MoEMethodTPU(fp8_config, fp8_config.get_moe_config(layer))
            # [ml3-patch] compressed-tensors stores FP8 block scales as `weight_scale` (not the DeepSeek-style
            # `weight_scale_inv`); register the MoE scale params under the checkpoint name so load_weights finds them.
            _ml3_method.weight_scale_name = "weight_scale"
            return _ml3_method
""",
    ),
]

# Opt-in Eagle draft-model patches 2-5 (ML3_EAGLE_PATCHES=1, default off); see the module docstring.
EAGLE_PATCHES = [
    (
        pathlib.Path("/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/mistral_large_3_eagle.py"),
        """        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )""",
        """        # [ml3-patch] EagleMistralLarge3Model calls nn.Module.__init__ instead of DeepseekV2Model.__init__,
        # leaving self.is_fused_shared_expert_enabled unset when DeepseekV2Model.load_weights runs.
        # Also expose EAGLE-1/2 single-hidden-state attributes expected by vllm_torchtpu's Eagle3Proposer.
        self.is_fused_shared_expert_enabled = False
        self.use_aux_hidden_state = False
        self.fc_input_size = config.hidden_size
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )""",
    ),
    (
        pathlib.Path("/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/mistral_large_3_eagle.py"),
        """        # draft model quantization config may differ from target model
        self.quant_config = VllmConfig.get_quantization_config(
            speculative_config.draft_model_config, vllm_config.load_config
        )
        vllm_config.quant_config = self.quant_config""",
        """        # [ml3-patch-quant] Route Eagle draft FP8 quantization through TPU VllmFp8Config
        # instead of upstream CUDA Fp8Config (which invokes torch.ops._C.static_scaled_fp8_quant).
        from vllm_torchtpu.layers.adapter.quantization import get_tpu_quantization_config
        self.quant_config = get_tpu_quantization_config(vllm_config)
        vllm_config.quant_config = self.quant_config""",
    ),
    (
        pathlib.Path("/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/mistral_large_3_eagle.py"),
        """    def get_language_model(self) -> torch.nn.Module:
        return self.model""",
        """    def get_language_model(self) -> torch.nn.Module:
        return self.model

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states""",
    ),
    (
        ROOT / "layers/adapter/custom_ops/mla_attention_op.py",
        """        flipped = getattr(self.kv_b_proj, WEIGHT_FLIPPED_ATTR, False)
        kv_b_weight = None
        if flipped:
            kv_b_weight = cast(torch.Tensor, self.kv_b_proj.weight)
            self.kv_b_proj.weight = Parameter(
                kv_b_weight.transpose(0, 1).contiguous(), requires_grad=False
            )
        try:
            super().process_weights_after_loading(act_dtype)
        finally:
            if flipped:
                assert kv_b_weight is not None
                self.kv_b_proj.weight = Parameter(kv_b_weight, requires_grad=False)""",
        """        # [ml3-patch-kvb] Upstream get_and_maybe_dequant_weights transposes dequant_weights when
        # `not quant_method.block_quant` (per-tensor FP8, e.g. Mistral-Large-3-Eagle kv_b_proj).
        # Temporarily mark block_quant=True while weight is un-flipped to [n_out, n_in] so it is not double-transposed.
        flipped = getattr(self.kv_b_proj, WEIGHT_FLIPPED_ATTR, False)
        kv_b_weight = None
        _qm = getattr(self.kv_b_proj, "quant_method", None)
        _restore_bq = flipped and _qm is not None and getattr(_qm, "block_quant", True) is False
        if flipped:
            kv_b_weight = cast(torch.Tensor, self.kv_b_proj.weight)
            self.kv_b_proj.weight = Parameter(
                kv_b_weight.transpose(0, 1).contiguous(), requires_grad=False
            )
            if _restore_bq:
                _qm.block_quant = True
        try:
            super().process_weights_after_loading(act_dtype)
        finally:
            if _restore_bq:
                _qm.block_quant = False
            if flipped:
                assert kv_b_weight is not None
                self.kv_b_proj.weight = Parameter(kv_b_weight, requires_grad=False)""",
    ),
]


def main() -> int:
    patches = list(PATCHES)
    if os.environ.get("ML3_EAGLE_PATCHES", "0") == "1":
        patches += EAGLE_PATCHES
    else:
        print("[ml3-patch] Eagle draft-model patches 2-5 skipped (set ML3_EAGLE_PATCHES=1 to apply)")
    rc = 0
    for path, old, new in patches:
        src = path.read_text()
        if new in src:
            print(f"[ml3-patch] already applied: {path}")
            continue
        if src.count(old) != 1:
            print(f"[ml3-patch] ERROR: anchor not found exactly once in {path}", file=sys.stderr)
            rc = 1
            continue
        path.write_text(src.replace(old, new))
        py_compile.compile(str(path), doraise=True)
        print(f"[ml3-patch] applied: {path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
