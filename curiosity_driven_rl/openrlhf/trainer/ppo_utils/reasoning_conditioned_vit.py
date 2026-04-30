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
    4. Gates the output with a zero-initialized scalar
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

        # ── ZERO-INIT GATE ──
        # Scalar initialized to 0. Controls adapter contribution.
        # At t=0: tanh(0)=0 → adapter does nothing → frozen ViT behavior
        self.gate = nn.Parameter(torch.zeros(1))

        # Layer norm for the adapter input (stabilizes training)
        self.adapter_norm = nn.LayerNorm(d_visual)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.adapter_k_proj.weight, gain=0.01)
        nn.init.xavier_uniform_(self.adapter_v_proj.weight, gain=0.01)
        nn.init.zeros_(self.adapter_out_proj.weight)

    def forward(
        self,
        visual_features: torch.Tensor,        # (N_patches, d_visual) — packed, no batch dim
        adapter_features: torch.Tensor,        # (1, K_adapt, d_reasoning) — the reasoning prompts
        original_block_output: torch.Tensor,   # (N_patches, d_visual) — output of frozen block
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
        attn_weights = torch.matmul(Q, adapter_K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn_weights: (1, n_heads, N, K_adapt)

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Weighted sum of adapter values
        adapter_output = torch.matmul(attn_weights, adapter_V)
        # adapter_output: (1, n_heads, N, head_dim)

        adapter_output = adapter_output.transpose(1, 2).contiguous().view(1, N, D)
        adapter_output = self.adapter_out_proj(adapter_output)
        adapter_output = adapter_output.squeeze(0)  # (N, D)

        # ── ZERO GATING ──
        # original_block_output is already the full block output (self-attn + MLP).
        # We add the gated adapter contribution as a residual.
        return original_block_output + torch.tanh(self.gate) * adapter_output


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

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        reasoning_input_ids: Optional[torch.Tensor] = None,
        reasoning_attention_mask: Optional[torch.Tensor] = None,
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

        # Step 2: Frozen ViT preprocessing — replicate the real forward exactly
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

        fullatt_block_indexes = set(self.vit.fullatt_block_indexes)

        # Step 3: Block loop with adapter injection
        for idx, block in enumerate(self.vit.blocks):
            pre_block_features = (
                x.clone() if idx in self.inject_at and adapter_prompts is not None else None
            )

            this_cu_seqlens = cu_seqlens if idx in fullatt_block_indexes else cu_window_seqlens

            with torch.no_grad():
                x = block(x, cu_seqlens=this_cu_seqlens, position_embeddings=position_embeddings)

            if idx in self.inject_at and adapter_prompts is not None:
                x = self.adapter_layers[str(idx)](
                    visual_features=pre_block_features,
                    adapter_features=adapter_prompts,
                    original_block_output=x,
                )

        # Step 4: Merger + reverse window ordering (frozen)
        with torch.no_grad():
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
            print(f"[ConditionedViT] Block {name:>2s} gate: {torch.tanh(layer.gate).item():.6f}")


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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, N_merged, d_merged) → (B, 3, H, W)"""
        B, N, D = features.shape
        n = self.n_patches_per_side

        patches = self.proj(features)
        patches = patches.view(B, n, n, self.patch_size, self.patch_size, 3)
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = patches.view(B, 3, self.image_size, self.image_size)
        img = self.refine(img)
        return img

    def to_pil(self, features: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            img_tensor = self.forward(features.unsqueeze(0))
            img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
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

        # Decode to PIL image
        result_image = self.decoder.to_pil(conditioned_features)
        result_image = result_image.resize(image.size, Image.LANCZOS)

        return result_image

    def _preprocess_image(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess PIL image using the Qwen2.5-VL processor.
        Returns (pixel_values, grid_thw) matching ViT input format.
        """
        from qwen_vl_utils import process_vision_info

        messages = [{"role": "user", "content": [{"type": "image", "image": image}]}]
        image_inputs, _ = process_vision_info(messages)

        inputs = self.processor(
            images=image_inputs,
            return_tensors="pt",
        )
        pixel_values = inputs['pixel_values'].to(self.device, dtype=torch.bfloat16)
        grid_thw = inputs['image_grid_thw'].to(self.device)
        return pixel_values, grid_thw

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
    print(f"  Gate:         {torch.tanh(adapter.gate).item():.6f}")

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
