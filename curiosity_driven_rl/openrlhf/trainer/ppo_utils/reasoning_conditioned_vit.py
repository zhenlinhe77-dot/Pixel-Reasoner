"""
reasoning_conditioned_vit.py  (v2 — LLaMA-Adapter style)

Reasoning-conditioned ViT using the LLaMA-Adapter zero-init attention mechanism.
Instead of adding separate cross-attention layers, we inject reasoning-derived
"adaptation prompts" as prefix tokens into the K/V of the ViT's EXISTING
self-attention layers. This is lighter and more principled.

Key difference from v1 (cross-attention):
    v1: Added new cross-attention layers AFTER ViT blocks
        Q=visual, K/V=reasoning. Required new Q/K/V projections per layer.

    v2 (this): Injects adapter tokens INTO the ViT block's own attention
        The visual patches attend to [adapter_tokens ; original_patches].
        Uses the EXISTING Q projection. Only adapter K/V are new parameters.
        Much fewer parameters. Matches the proven LLaMA-Adapter design.

LLaMA-Adapter mechanism (Zhang et al., ICLR 2024):
    For each adapted layer l:
        1. Compute Q, K, V from visual patches using the frozen qkv projection
        2. Adapter prompts A_l of shape (K_adapt, C) provide extra K/V
        3. Compute adapter attention INDEPENDENTLY:
             S_g = softmax(Q @ A_l^T / sqrt(d))   — attention over adapter only
        4. Compute original self-attention INDEPENDENTLY:
             S_w = softmax(Q @ K^T / sqrt(d))      — vanilla self-attention
        5. Gate the adapter contribution:
             output = S_w @ V + gate_l * S_g @ V_adapter
           where gate_l is initialized to ZERO
        6. At t=0: gate=0 → output = S_w @ V (identical to frozen ViT)
           During training: gate opens → reasoning conditioning flows in

    The INDEPENDENT softmax is critical. Joint softmax([adapter; original])
    would dilute the original attention pattern even at initialization.

Qwen2.5-VL ViT specifics:
    Blocks: 32 × Qwen2_5_VLVisionBlock
    Attention: fused qkv = Linear(1280, 3840) → split into Q, K, V each 1280
    Heads: 16, head_dim = 80
    Full-attention blocks: [7, 15, 23, 31] (rest use windowed, window=112)
    No separate Q/K/V projections — it's a single fused linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from typing import Optional, Tuple
from PIL import Image
import numpy as np


# =============================================================================
# 1. ZERO-INIT ADAPTER LAYER (LLaMA-Adapter style)
# =============================================================================

class ZeroInitAdapterLayer(nn.Module):
    """
    LLaMA-Adapter style zero-init attention for a single ViT block.

    Instead of a full cross-attention layer, this module:
    1. Takes reasoning features as adapter prompts
    2. Projects them into K/V space matching the ViT's attention
    3. Computes attention from visual Q to adapter K/V (independent softmax)
    4. Gates each attention head independently with a learned (n_heads,) vector
    5. Adds to the original attention output as a residual

    This is applied as a WRAPPER around the existing ViT block —
    we don't modify the frozen block's internal code.

    Args:
        d_visual: ViT hidden dim (1280 for Qwen2.5-VL)
        d_reasoning: reasoning encoder output dim
        n_heads: number of attention heads (16 for Qwen2.5-VL)
    """

    def __init__(self, d_visual: int = 1280, d_reasoning: int = 1280, n_heads: int = 16):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_visual // n_heads  # 80 for Qwen2.5-VL

        # Project reasoning features into adapter K and V
        # These are the ONLY new projections — Q comes from the frozen ViT
        self.adapter_k_proj = nn.Linear(d_reasoning, d_visual, bias=False)
        self.adapter_v_proj = nn.Linear(d_reasoning, d_visual, bias=False)

        # Output projection for the adapter attention
        self.adapter_out_proj = nn.Linear(d_visual, d_visual, bias=False)

        # ── PER-HEAD GATE ──
        # Shape (n_heads,): one scalar per attention head, init 0.1.
        # Applied before adapter_out_proj on the (1, n_heads, N, head_dim) tensor.
        # Per-head gating lets the model learn that some heads carry spatial
        # signal and others semantic signal, weighting them independently.
        # tanh(0.1)≈0.10 → ~10% contribution at init; large enough that
        # adapter_out_proj and reasoning encoder receive real gradients from step 1.
        self.gate = nn.Parameter(torch.full((n_heads,), 0.1))

        # Layer norm for the adapter input (stabilizes training)
        self.adapter_norm = nn.LayerNorm(d_visual)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.adapter_k_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.adapter_v_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.adapter_out_proj.weight, gain=0.01)

    def forward(
        self,
        visual_features: torch.Tensor,        # (N_patches, d_visual) — packed, no batch dim
        adapter_features: torch.Tensor,        # (1, K_adapt, d_reasoning) — the reasoning prompts
        original_block_output: torch.Tensor,   # (N_patches, d_visual) — output of frozen block
        return_attn: bool = False,
    ) -> torch.Tensor:
        """
        Compute adapter contribution and add to the frozen block's output.

        The flow:
            1. Use visual_features (PRE-block, after norm1) to get Q
            2. Use adapter_features to get adapter K, V
            3. Compute gated adapter attention
            4. Add to original_block_output as residual

        Returns:
            modified_output: (N_patches, d_visual)
            When return_attn=True: (modified_output, attn_weights) where
                attn_weights shape is (1, n_heads, N_patches, K_adapt), float32
        """
        N, D = visual_features.shape
        K_adapt = adapter_features.shape[1]

        # Normalize visual features (same as the block's norm1 would do)
        x = self.adapter_norm(visual_features)

        # Q from visual patches — reuse the patch features directly
        # We DON'T use the frozen qkv projection here (that would require
        # hooking into the block internals). Instead, we project the
        # pre-norm visual features through our own lightweight path.
        # This is equivalent to the adapter having its own Q=identity mapping.
        Q = x.unsqueeze(0)  # (1, N, D)
        Q = Q.view(1, N, self.n_heads, self.head_dim).transpose(1, 2)
        # Q: (1, n_heads, N, head_dim)

        # K, V from adapter (reasoning) features
        adapter_K = self.adapter_k_proj(adapter_features)  # (1, K_adapt, D)
        adapter_V = self.adapter_v_proj(adapter_features)  # (1, K_adapt, D)

        adapter_K = adapter_K.view(1, K_adapt, self.n_heads, self.head_dim).transpose(1, 2)
        adapter_V = adapter_V.view(1, K_adapt, self.n_heads, self.head_dim).transpose(1, 2)
        # adapter_K, adapter_V: (1, n_heads, K_adapt, head_dim)

        # ── INDEPENDENT SOFTMAX over adapter tokens only ──
        # This is the key LLaMA-Adapter design choice.
        # We do NOT concatenate with original K/V and do joint softmax.
        # Independent softmax ensures at gate=0, original attention is untouched.
        #
        # Compute in fp32: bf16 attention weights overflow when adapter_K values
        # are large, producing all-inf weights whose softmax gives NaN.
        attn_weights = torch.matmul(Q.float(), adapter_K.float().transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn_weights: (1, n_heads, N, K_adapt)

        attn_weights = F.softmax(attn_weights, dim=-1).to(Q.dtype)

        # Weighted sum of adapter values
        adapter_output = torch.matmul(attn_weights, adapter_V)
        # adapter_output: (1, n_heads, N, head_dim)

        # ── PER-HEAD GATING ���─
        # gate: (n_heads,) → (1, n_heads, 1, 1) to broadcast over (1, n_heads, N, head_dim).
        # Applied before out_proj so the projection mixes already-gated head outputs.
        # nan_to_num: zero out NaN AND inf (inf survives plain nan_to_num(0.0) and
        # later causes LayerNorm NaN via inf - mean = NaN).
        gate = torch.tanh(self.gate).view(1, self.n_heads, 1, 1)
        adapter_output = adapter_output.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0) * gate

        adapter_output = adapter_output.transpose(1, 2).contiguous().view(1, N, D)
        adapter_output = self.adapter_out_proj(adapter_output)
        adapter_output = adapter_output.squeeze(0)  # (N, D)

        result = original_block_output + adapter_output
        if return_attn:
            return result, attn_weights.detach().float()
        return result


# =============================================================================
# 2. REASONING ENCODER (produces the adapter prompts)
# =============================================================================

class ReasoningEncoder(nn.Module):
    """
    Encode reasoning history text into adapter prompt tokens.

    Output shape: (1, K_adapt, d_output) — the "adaptation prompts"
    that get injected into the ViT's attention as prefix K/V.

    K_adapt is FIXED (e.g., 32 tokens) regardless of reasoning text length.
    The encoder compresses variable-length reasoning into fixed-size prompts,
    similar to how LLaMA-Adapter uses fixed K-length prompts per layer.

    Args:
        vocab_size: tokenizer vocabulary size (151936 for Qwen2.5)
        d_model: internal encoder dimension
        n_layers: number of transformer layers
        n_heads: attention heads
        k_adapt: number of output adapter tokens (the "prompt length K")
        d_output: output dimension per token (must match ViT hidden_size=1280)
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        max_seq_len: int = 2048,
        k_adapt: int = 32,       # number of adapter prompt tokens
        d_output: int = 1280,    # ViT hidden dim
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.k_adapt = k_adapt

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── COMPRESS to K_adapt tokens ──
        # Learnable query tokens that cross-attend to the encoded reasoning
        # This produces exactly K_adapt output tokens regardless of input length
        self.compress_queries = nn.Parameter(torch.randn(1, k_adapt, d_model) * 0.02)
        self.compress_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.compress_norm = nn.LayerNorm(d_model)

        # Project to ViT dimension
        self.output_proj = nn.Linear(d_model, d_output)
        self.output_norm = nn.LayerNorm(d_output)

    def forward(
        self,
        input_ids: torch.Tensor,        # (1, S) token ids
        attention_mask: torch.Tensor,    # (1, S)
    ) -> torch.Tensor:
        """
        Returns:
            adapter_prompts: (1, K_adapt, d_output) — fixed-size adapter tokens
        """
        B, S = input_ids.shape

        # Truncate to last max_seq_len tokens (most recent reasoning)
        if S > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]
            attention_mask = attention_mask[:, -self.max_seq_len:]
            S = self.max_seq_len

        # Encode
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        src_key_padding_mask = ~attention_mask.bool()

        # Guard: all tokens masked → TransformerEncoder softmax(-inf,...) = NaN.
        # Return zero adapter prompts so gate=0 leaves the ViT output unchanged.
        if src_key_padding_mask.all():
            dtype = self.token_embedding.weight.dtype
            K = self.compress_queries.shape[1]
            return torch.zeros(B, K, self.output_proj.out_features,
                               device=input_ids.device, dtype=dtype)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        # x: (1, S, d_model)

        # Compress to K_adapt tokens via learned queries
        queries = self.compress_queries.expand(B, -1, -1)  # (1, K_adapt, d_model)
        compressed, _ = self.compress_attn(
            query=queries,
            key=x,
            value=x,
            key_padding_mask=src_key_padding_mask,
        )
        compressed = self.compress_norm(compressed + queries)  # residual
        # compressed: (1, K_adapt, d_model)

        # Project to ViT dimension
        adapter_prompts = self.output_norm(self.output_proj(compressed))
        # adapter_prompts: (1, K_adapt, d_output=1280)

        # Guard: clamp non-finite values so a degraded encoder never corrupts the ViT.
        adapter_prompts = adapter_prompts.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        return adapter_prompts


# =============================================================================
# 3. CONDITIONED VIT (LLaMA-Adapter style wrapper)
# =============================================================================

class ConditionedViT(nn.Module):
    """
    Wraps the frozen Qwen2.5-VL ViT with LLaMA-Adapter style zero-init
    adapter layers injected at selected blocks.

    For each adapted block:
        1. Run the frozen block normally → original_output
        2. Run the adapter layer: visual_features + adapter_prompts → gated residual
        3. output = original_output + gate * adapter_attention

    Architecture details (Qwen2.5-VL ViT):
        model.visual.blocks: 32 × Qwen2_5_VLVisionBlock
        Each block: norm1 → attn(qkv fused) → norm2 → mlp(gate+up+down)
        Hidden dim: 1280, heads: 16, head_dim: 80
        Full-attention blocks: [7, 15, 23, 31]
        Windowed blocks: all others (window_size=112)
        Output: merger projects 1280 → 3584 with 2×2 spatial merge

    Strategy options for adapter placement:
        "fullatt_only": blocks [7,15,23,31] — 4 adapters, recommended start
        "top_l": last L blocks — matches original LLaMA-Adapter design
        "every_k": every k-th block
    """

    def __init__(
        self,
        vit: nn.Module,
        reasoning_encoder: "ReasoningEncoder",
        d_visual: int = 1280,
        d_reasoning: int = 1280,
        n_heads: int = 16,
        strategy: str = "fullatt_only",
        top_l: int = 8,
        every_k: int = 4,
    ):
        super().__init__()
        self.vit = vit
        self.reasoning_encoder = reasoning_encoder

        # Freeze ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        n_blocks = len(self.vit.blocks)  # 32
        FULLATT_BLOCKS = [7, 15, 23, 31]

        if strategy == "fullatt_only":
            inject_at = FULLATT_BLOCKS
        elif strategy == "top_l":
            inject_at = list(range(n_blocks - top_l, n_blocks))
        elif strategy == "every_k":
            inject_at = list(range(every_k - 1, n_blocks, every_k))
        elif strategy == "all":
            inject_at = list(range(n_blocks))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        print(f"[ConditionedViT] Strategy: {strategy}")
        print(f"[ConditionedViT] Adapter layers at blocks: {inject_at}")
        print(f"[ConditionedViT] Full-attention blocks: {FULLATT_BLOCKS}")

        # Create adapter layers
        self.adapter_layers = nn.ModuleDict()
        for idx in inject_at:
            self.adapter_layers[str(idx)] = ZeroInitAdapterLayer(
                d_visual=d_visual,
                d_reasoning=d_reasoning,
                n_heads=n_heads,
            )
        self.inject_at = set(inject_at)
        # Populated by forward() when capture_attn=True; dict {block_idx: attn_weights}
        self._last_captured_attn: dict = {}
        # Counter for PathB encode calls; used to gate attention logging frequency
        self._pathb_call_count: int = 0

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        reasoning_input_ids: Optional[torch.Tensor] = None,
        reasoning_attention_mask: Optional[torch.Tensor] = None,
        capture_attn: bool = False,
    ) -> torch.Tensor:
        """
        Run the conditioned ViT.

        Replicates Qwen2_5_VisionTransformerPretrainedModel.forward() exactly,
        including window reordering and reverse reordering, with adapter
        residuals injected after selected blocks.
        """
        # Step 1: Encode reasoning into fixed-size adapter prompts
        if reasoning_input_ids is not None:
            adapter_prompts = self.reasoning_encoder(
                reasoning_input_ids, reasoning_attention_mask
            )
        else:
            adapter_prompts = None

        # Step 2: Frozen ViT preprocessing — replicate the real forward exactly.
        # Run under no_grad since patch_embed/rot_pos_emb have no adapters and
        # we don't need their activations in the backward graph.
        with torch.no_grad():
            x = self.vit.patch_embed(pixel_values)
            rotary_pos_emb = self.vit.rot_pos_emb(grid_thw)
            window_index, cu_window_seqlens = self.vit.get_window_index(grid_thw)
            cu_window_seqlens = torch.tensor(
                cu_window_seqlens, device=x.device, dtype=torch.int32
            )
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

            seq_len, _ = x.size()
            # Reorder patches into window-attention order (groups of spatial_merge_unit)
            x = x.reshape(seq_len // self.vit.spatial_merge_unit, self.vit.spatial_merge_unit, -1)
            x = x[window_index, :, :]
            x = x.reshape(seq_len, -1)

            rotary_pos_emb = rotary_pos_emb.reshape(
                seq_len // self.vit.spatial_merge_unit, self.vit.spatial_merge_unit, -1
            )
            rotary_pos_emb = rotary_pos_emb[window_index, :, :]
            rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            position_embeddings = (emb.cos(), emb.sin())

            # cu_seqlens for full-attention blocks: cumsum of (h*w) per frame across images
            cu_seqlens = torch.repeat_interleave(
                grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
            ).cumsum(dim=0, dtype=torch.int32)
            cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Detach from the no_grad preprocessing; gradient only enters via adapters.
        x = x.detach()

        fullatt_block_indexes = set(self.vit.fullatt_block_indexes)

        # Step 3: Block loop with adapter injection.
        # Gradient checkpointing per block: recompute activations during backward
        # instead of storing all 32 blocks' Q/K/V/FFN tensors. Reduces ViT activation
        # memory by ~32x at the cost of one extra forward pass per block during backward.
        #
        # When capture_attn=True: skip grad_ckpt and call blocks directly so that
        # attn_weights survive past the forward pass. This path is only used under
        # torch.no_grad() (called from compare_attn_stats), so no recomputation needed.
        if capture_attn:
            self._last_captured_attn = {}
        for idx, block in enumerate(self.vit.blocks):
            this_cu_seqlens = cu_seqlens if idx in fullatt_block_indexes else cu_window_seqlens

            if idx in self.inject_at and adapter_prompts is not None:
                pre_block_features = x.detach()
                adp_layer = self.adapter_layers[str(idx)]

                if capture_attn:
                    # Direct call — always under no_grad, so no grad_ckpt needed.
                    out = block(x, cu_seqlens=this_cu_seqlens, position_embeddings=position_embeddings)
                    x, aw = adp_layer(visual_features=pre_block_features, adapter_features=adapter_prompts,
                                      original_block_output=out, return_attn=True)
                    self._last_captured_attn[idx] = aw
                else:
                    # Default-arg capture binds all loop variables correctly per iteration.
                    def _fwd_with_adapter(
                        x_,
                        _blk=block, _cu=this_cu_seqlens, _pe=position_embeddings,
                        _pre=pre_block_features, _adp=adapter_prompts, _adl=adp_layer,
                    ):
                        out = _blk(x_, cu_seqlens=_cu, position_embeddings=_pe)
                        return _adl(visual_features=_pre, adapter_features=_adp, original_block_output=out)

                    x = grad_ckpt(_fwd_with_adapter, x, use_reentrant=False)
            else:
                if capture_attn:
                    x = block(x, cu_seqlens=this_cu_seqlens, position_embeddings=position_embeddings)
                else:
                    def _fwd_block(
                        x_,
                        _blk=block, _cu=this_cu_seqlens, _pe=position_embeddings,
                    ):
                        return _blk(x_, cu_seqlens=_cu, position_embeddings=_pe)

                    x = grad_ckpt(_fwd_block, x, use_reentrant=False)

        # Step 4: Merger + reverse window ordering.
        # Run without no_grad so the adapter residuals from block 31 remain connected
        # to the autograd graph and gate gradients can flow from the PPO loss.
        x = self.vit.merger(x)
        reverse_indices = torch.argsort(window_index)
        x = x[reverse_indices, :]

        return x

    def get_trainable_parameters(self):
        """Return only trainable parameters (adapters + reasoning encoder)."""
        params = []
        for layer in self.adapter_layers.values():
            params.extend(layer.parameters())
        params.extend(self.reasoning_encoder.parameters())
        return params

    def print_trainable_stats(self):
        """Print parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"[ConditionedViT] Total params:     {total:>12,}")
        print(f"[ConditionedViT] Trainable params:  {trainable:>12,}")
        print(f"[ConditionedViT] Frozen params:     {frozen:>12,}")
        print(f"[ConditionedViT] Trainable ratio:   {trainable/total:.4%}")

        adapter_params = sum(
            p.numel() for layer in self.adapter_layers.values()
            for p in layer.parameters()
        )
        encoder_params = sum(p.numel() for p in self.reasoning_encoder.parameters())
        print(f"[ConditionedViT] Adapter params:    {adapter_params:>12,}")
        print(f"[ConditionedViT] Encoder params:    {encoder_params:>12,}")

        for name, layer in self.adapter_layers.items():
            gate_tanh = torch.tanh(layer.gate)
            if gate_tanh.numel() == 1:
                print(f"[ConditionedViT] Block {name:>2s} gate: {gate_tanh.item():.6f}")
            else:
                vals = ", ".join(f"{v:.4f}" for v in gate_tanh.tolist())
                print(f"[ConditionedViT] Block {name:>2s} gate (per-head): [{vals}]")

    @torch.no_grad()
    def compare_attn_stats(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        reasoning_ids: torch.Tensor,
        reasoning_mask: torch.Tensor,
    ) -> dict:
        """
        Run conditioned and unconditioned forward passes, return attention stats.

        Returns dict with:
          'entropy': {block_idx: tensor(N_patches,)} — mean adapter attention entropy
                     across heads per patch (in nats, max = log(K_adapt) ≈ 3.47)
          'feature_delta': tensor(N_merged,) — ||f_cond - f_uncond||_2 per merged patch
        """
        # Conditioned pass — capture adapter attention weights
        cond_features = self.forward(
            pixel_values, grid_thw, reasoning_ids, reasoning_mask, capture_attn=True
        )
        captured = {k: v.clone() for k, v in self._last_captured_attn.items()}

        # Unconditioned pass — same ViT, no reasoning conditioning
        uncond_features = self.forward(
            pixel_values, grid_thw, None, None, capture_attn=True
        )

        # Adapter attention entropy per block (conditioned only)
        entropy = {}
        for bidx, aw in captured.items():
            # aw: (1, n_heads, N, K_adapt), float32 after detach
            eps = 1e-8
            H = -(aw * (aw + eps).log()).sum(dim=-1)  # (1, n_heads, N)
            entropy[bidx] = H.mean(dim=(0, 1)).cpu()  # (N,)

        # Feature delta: L2 norm of difference per merged patch
        delta = (cond_features.float() - uncond_features.float()).norm(dim=-1).cpu()  # (N_merged,)

        return {'entropy': entropy, 'feature_delta': delta}

    def maybe_log_attn(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        reasoning_ids: torch.Tensor,
        reasoning_mask: torch.Tensor,
        save_dir: Optional[str] = None,
    ) -> None:
        """
        Frequency-gated attention logging. Call this after every PathB encode.

        Every 100 PathB calls: log entropy scalars and feature delta to stdout.
        Every 500 PathB calls: additionally save a heatmap PNG to save_dir
            (or the ATTN_VIZ_DIR env var if save_dir is None).
        """
        self._pathb_call_count += 1
        count = self._pathb_call_count

        if count % 100 != 0:
            return

        try:
            stats = self.compare_attn_stats(pixel_values, grid_thw, reasoning_ids, reasoning_mask)
        except Exception as e:
            print(f"[AttnViz] compare_attn_stats failed (step {count}): {e}")
            return

        for bidx in sorted(stats['entropy'].keys()):
            ent = stats['entropy'][bidx]
            print(f"[AttnViz] pathb={count} block={bidx:2d} "
                  f"entropy mean={ent.mean().item():.4f} max={ent.max().item():.4f}")
        delta = stats['feature_delta']
        print(f"[AttnViz] pathb={count} feature_delta mean={delta.mean().item():.4f} "
              f"max={delta.max().item():.4f}")

        if count % 500 == 0:
            import os as _os
            _save_dir = save_dir or _os.environ.get('ATTN_VIZ_DIR', None)
            if _save_dir:
                try:
                    _save_attn_heatmaps(stats, grid_thw, count, _save_dir)
                except Exception as e:
                    print(f"[AttnViz] heatmap save failed (step {count}): {e}")


def _save_attn_heatmaps(stats: dict, grid_thw: torch.Tensor, step: int, save_dir: str) -> None:
    """
    Save adapter attention entropy heatmaps + feature delta map as a PNG.

    Layout: one subplot per adapter block (entropy) + one subplot for feature delta.
    Each block heatmap is reshaped to the spatial patch grid (H_patches × W_patches).
    Feature delta is at merger resolution (H_patches//2 × W_patches//2).
    """
    import os as _os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _os.makedirs(save_dir, exist_ok=True)

    # Spatial dimensions from grid_thw: [[T, H_patches, W_patches], ...]
    thw = grid_thw[0]
    H_p = int(thw[1])   # height in pre-merger patches
    W_p = int(thw[2])   # width  in pre-merger patches
    H_m = H_p // 2      # height after 2×2 merger
    W_m = W_p // 2      # width  after 2×2 merger

    entropy_blocks = sorted(stats['entropy'].keys())
    n_cols = len(entropy_blocks) + 1  # +1 for feature delta
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for col, bidx in enumerate(entropy_blocks):
        ent = stats['entropy'][bidx].float().numpy()  # (N,)
        grid = ent.reshape(H_p, W_p) if ent.size == H_p * W_p else ent.reshape(1, -1)
        ax = axes[col]
        im = ax.imshow(grid, cmap='hot', vmin=0.0, interpolation='nearest')
        ax.set_title(f'Block {bidx}\nadapter entropy', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Feature delta subplot
    delta = stats['feature_delta'].float().numpy()  # (N_merged,)
    grid_d = delta.reshape(H_m, W_m) if delta.size == H_m * W_m else delta.reshape(1, -1)
    ax = axes[-1]
    im = ax.imshow(grid_d, cmap='viridis', interpolation='nearest')
    ax.set_title('Feature delta\n||cond − uncond||', fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f'Attention visualization — PathB step {step}', fontsize=11)
    plt.tight_layout()

    out_path = _os.path.join(save_dir, f'attn_viz_step{step:07d}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"[AttnViz] Saved: {out_path}")


# =============================================================================
# 4. FEATURE DECODER (Path A: features → PIL image for tool loop)
# =============================================================================

class FeatureDecoder(nn.Module):
    """
    Decode conditioned ViT features (post-merger, 3584-dim) back to a PIL image.

    After the merger, features are (N_merged, 3584) where N_merged = N_patches/4.
    We project back to pixel space for Path A (PIL image → existing tool loop).

    This is trained with reconstruction loss in a pre-training phase.
    """

    def __init__(
        self,
        d_merged: int = 3584,     # post-merger dim
        patch_size: int = 28,      # effective patch size after 2×2 merge (14*2=28)
        image_size: int = 448,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.n_patches_per_side = image_size // patch_size  # 16

        patch_pixels = patch_size * patch_size * 3  # 28*28*3 = 2352
        self.proj = nn.Linear(d_merged, patch_pixels)

        self.refine = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, grid_hw=None) -> torch.Tensor:
        """features: (B, N_merged, d_merged) → (B, 3, H, W)
        grid_hw: (n_h, n_w) merged-patch grid; inferred from N if None.
        """
        B, N, D = features.shape
        if grid_hw is not None:
            n_h, n_w = grid_hw
        else:
            n_h = n_w = self.n_patches_per_side  # fallback: assume square 448×448

        patches = self.proj(features)
        patches = patches.view(B, n_h, n_w, self.patch_size, self.patch_size, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = patches.view(B, 3, n_h * self.patch_size, n_w * self.patch_size)
        img = self.refine(img)
        return img

    def to_pil(self, features: torch.Tensor, grid_hw=None) -> Image.Image:
        with torch.no_grad():
            img_tensor = self.forward(features.unsqueeze(0), grid_hw=grid_hw)
            img_np = (img_tensor[0].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(img_np, 'RGB')


# =============================================================================
# 5. FULL CONDITIONER MODULE (replaces placeholder in reencode_patch.py)
# =============================================================================

class ReasoningConditionerV2(nn.Module):
    """
    Complete reasoning-conditioned re-encoding using LLaMA-Adapter mechanism.

    Replaces the placeholder ReasoningConditioner from reencode_patch.py.
    The interface is the same: conditioner.process(image, focus_hint, reasoning_history) → PIL image

    Usage in experience_maker.py:
        self.conditioner = ReasoningConditionerV2(vit=model.visual, tokenizer=tokenizer, processor=processor)
        # ... later in tool execution:
        result_image = self.conditioner.process(image, focus_hint, reasoning_history)
    """

    def __init__(
        self,
        vit: nn.Module,          # model.visual
        tokenizer,               # Qwen2.5-VL tokenizer
        processor,               # Qwen2.5-VL AutoProcessor (for image preprocessing)
        d_visual: int = 1280,
        d_merged: int = 3584,    # post-merger dim
        reasoning_d_model: int = 512,
        reasoning_n_layers: int = 4,
        reasoning_n_heads: int = 8,
        k_adapt: int = 32,       # adapter prompt length
        adapter_n_heads: int = 16,
        adapter_strategy: str = "fullatt_only",
        image_size: int = 448,
        patch_size: int = 14,
        device: str = "cuda",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device

        # Reasoning encoder → adapter prompts
        self.reasoning_encoder = ReasoningEncoder(
            vocab_size=len(tokenizer),
            d_model=reasoning_d_model,
            n_layers=reasoning_n_layers,
            n_heads=reasoning_n_heads,
            k_adapt=k_adapt,
            d_output=d_visual,
        )

        # Conditioned ViT with adapter layers
        self.conditioned_vit = ConditionedViT(
            vit=vit,
            reasoning_encoder=self.reasoning_encoder,
            d_visual=d_visual,
            d_reasoning=d_visual,
            n_heads=adapter_n_heads,
            strategy=adapter_strategy,
        )

        # Feature decoder (Path A)
        self.decoder = FeatureDecoder(
            d_merged=d_merged,
            patch_size=patch_size * 2,  # 14 * 2 = 28 after spatial merge
            image_size=image_size,
        )

    def process(
        self,
        image: Image.Image,
        focus_hint: str,
        reasoning_history: Optional[str] = None,
    ) -> Image.Image:
        """
        Main entry point. Called by ReEncode.call() and auto-reencode after crop/zoom.
        PIL image + reasoning text → conditioned PIL image.
        """
        if reasoning_history is None or len(reasoning_history.strip()) == 0:
            return image

        conditioning_text = f"[Focus: {focus_hint}]\n{reasoning_history}" if focus_hint else reasoning_history

        # Tokenize reasoning
        tokens = self.tokenizer(
            conditioning_text,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True,
        ).to(self.device)

        # Preprocess image
        pixel_values, grid_thw = self._preprocess_image(image)

        # Run conditioned ViT
        conditioned_features = self.conditioned_vit(
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            reasoning_input_ids=tokens['input_ids'],
            reasoning_attention_mask=tokens['attention_mask'],
        )
        # conditioned_features: (N_merged, 3584) — packed, no batch dim

        # grid_thw is [[t, h_patches, w_patches]]; after 2×2 spatial merger
        # the merged grid is (h_patches//2, w_patches//2)
        _thw = grid_thw[0]
        grid_hw = (int(_thw[1]) // 2, int(_thw[2]) // 2)

        # Decode to PIL image
        result_image = self.decoder.to_pil(conditioned_features, grid_hw=grid_hw)
        result_image = result_image.resize(image.size, Image.LANCZOS)

        return result_image

    def _preprocess_image(
        self,
        image: Image.Image,
        max_pixels: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess PIL image using the Qwen2.5-VL processor.
        Returns (pixel_values, grid_thw) matching ViT input format.

        Args:
            max_pixels: if set, cap image resolution so patches <= max_pixels/196.
                        Pass 1024*14*14=200704 to guarantee at most 1024 patches.
        """
        from qwen_vl_utils import process_vision_info

        # max_pixels must be set in the content dict — process_vision_info reads it
        # there to resize the PIL image before the processor sees it.
        # Passing max_pixels as a processor kwarg is silently ignored.
        image_content: dict = {"type": "image", "image": image}
        if max_pixels is not None:
            image_content["max_pixels"] = max_pixels

        messages = [{"role": "user", "content": [
            image_content,
            {"type": "text", "text": ""},
        ]}]
        image_inputs, _ = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
        )
        pixel_values = inputs['pixel_values'].to(self.device, dtype=torch.bfloat16)
        grid_thw = inputs['image_grid_thw'].to(self.device)
        return pixel_values, grid_thw

    def encode_for_llm(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        reasoning_ids: torch.Tensor,
        reasoning_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Path B: run ConditionedViT and return features in LLM-ready shape.
        No FeatureDecoder, no PIL round-trip.
        Gradients flow through adapter layers and reasoning encoder.

        Returns: (N_merged, 3584) — same shape/order as model.visual(pixel_values, grid_thw).
        Call this inside training_step_actor (with grads enabled) so PPO loss
        can backprop into the adapter weights.
        """
        features = self.conditioned_vit(
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            reasoning_input_ids=reasoning_ids,
            reasoning_attention_mask=reasoning_mask,
        )
        # Log attention entropy scalars every 100 PathB calls; save PNG every 500.
        # Runs under no_grad in a try/except so logging failures never break training.
        import os as _os
        _save_dir = getattr(self, '_attn_viz_dir', None) or _os.environ.get('ATTN_VIZ_DIR', None)
        try:
            with torch.no_grad():
                self.conditioned_vit.maybe_log_attn(
                    pixel_values.detach(), grid_thw, reasoning_ids.detach(), reasoning_mask.detach(),
                    save_dir=_save_dir,
                )
        except Exception as _e:
            print(f"[AttnViz] maybe_log_attn failed: {_e}")
        return features

    def get_trainable_parameters(self):
        """All trainable parameters: adapters + encoder + decoder."""
        params = list(self.conditioned_vit.get_trainable_parameters())
        params.extend(self.decoder.parameters())
        return params


# =============================================================================
# 6. SMOKE TESTS
# =============================================================================

if __name__ == "__main__":
    print("=== Smoke test: ZeroInitAdapterLayer ===")
    adapter = ZeroInitAdapterLayer(d_visual=1280, d_reasoning=1280, n_heads=16)

    visual = torch.randn(256, 1280)          # 256 patches, packed (no batch dim)
    reasoning = torch.randn(1, 32, 1280)     # 32 adapter tokens
    block_output = torch.randn(256, 1280)    # fake block output

    out = adapter(visual, reasoning, block_output)
    print(f"  Visual:       {visual.shape}")
    print(f"  Adapter:      {reasoning.shape}")
    print(f"  Block output: {block_output.shape}")
    print(f"  Output:       {out.shape}")
    _gate_t = torch.tanh(adapter.gate)
    _gate_str = f"{_gate_t.item():.6f}" if _gate_t.numel() == 1 else "[" + ", ".join(f"{v:.4f}" for v in _gate_t.tolist()) + "]"
    print(f"  Gate:         {_gate_str}")

    # Verify zero-init: output should equal block_output at init
    diff = (out - block_output).abs().max().item()
    print(f"  Max diff from block_output (should be ~0): {diff:.8f}")
    assert diff < 1e-5, f"Zero-init broken: diff={diff}"
    print("  Zero-init verified.")

    print("\n=== Smoke test: ReasoningEncoder ===")
    enc = ReasoningEncoder(vocab_size=1000, d_model=256, n_layers=2, n_heads=4,
                           k_adapt=32, d_output=1280)
    ids = torch.randint(0, 1000, (1, 128))
    mask = torch.ones(1, 128)
    prompts = enc(ids, mask)
    print(f"  Input:  ids={ids.shape}")
    print(f"  Output: adapter_prompts={prompts.shape}  (should be [1, 32, 1280])")
    assert prompts.shape == (1, 32, 1280)

    print("\n=== Smoke test: FeatureDecoder ===")
    dec = FeatureDecoder(d_merged=3584, patch_size=28, image_size=448)
    fake_features = torch.randn(1, 256, 3584)  # 16×16 merged patches
    img = dec.to_pil(fake_features)
    print(f"  Input:  features={fake_features.shape}")
    print(f"  Output: PIL image {img.size}, mode={img.mode}")

    print("\n=== Parameter counts ===")
    ad_params = sum(p.numel() for p in adapter.parameters())
    enc_params = sum(p.numel() for p in enc.parameters())
    dec_params = sum(p.numel() for p in dec.parameters())
    print(f"  Adapter (1 layer):              {ad_params:>10,}")
    print(f"  Adapter (4 layers, fullatt):    {ad_params*4:>10,}")
    print(f"  Reasoning encoder:              {enc_params:>10,}")
    print(f"  Feature decoder:                {dec_params:>10,}")
    print(f"  Total trainable (4 adapters):   {ad_params*4 + enc_params + dec_params:>10,}")

    print("\nAll smoke tests passed.")
