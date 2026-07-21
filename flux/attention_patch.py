import torch

def create_fmars_cross_attention_patch(reference_tokens, strength=1.0):
    """
    FLUX Cross-Attention (attn2) katmanı için FMARS patch'i.
    """
    if reference_tokens is None or strength <= 0.0:
        return None

    def attn2_patch_fn(q, k, v, extra_options, **kwargs):
        ref_tokens = reference_tokens.to(device=q.device, dtype=q.dtype)
        ref_kv = ref_tokens * strength
        
        # Shape matching
        if k.dim() == 4 and ref_kv.dim() == 3:
            batch_size, num_heads, seq_len, head_dim = k.shape
            ref_kv = ref_kv.view(1, -1, num_heads, head_dim)
            ref_kv = ref_kv.permute(0, 2, 1, 3)
            ref_kv = ref_kv.expand(batch_size, -1, -1, -1)
            cat_dim = 2
        elif k.dim() == 3 and ref_kv.dim() == 3:
            batch_size = k.shape[0]
            ref_kv = ref_kv.expand(batch_size, -1, -1)
            cat_dim = 1
        else:
            if ref_kv.shape[0] != k.shape[0]:
                expand_shape = [k.shape[0]] + [-1] * (ref_kv.dim() - 1)
                ref_kv = ref_kv.expand(*expand_shape)
            cat_dim = 2 if k.dim() == 4 else 1

        k_combined = torch.cat([k, ref_kv], dim=cat_dim)
        v_combined = torch.cat([v, ref_kv], dim=cat_dim)
        
        return {
            "q": q,
            "k": k_combined,
            "v": v_combined,
            "pe": kwargs.get("pe", None),
            "attn_mask": kwargs.get("attn_mask", None)
        }

    return attn2_patch_fn


def create_fmars_self_attention_patch(reference_tokens, strength=0.3):
    """
    FLUX Self-Attention (attn1) katmanı için FMARS patch'i.
    """
    if reference_tokens is None or strength <= 0.0:
        return None

    def attn1_patch_fn(q, k, v, extra_options, **kwargs):
        ref_tokens = reference_tokens.to(device=q.device, dtype=q.dtype)
        ref_kv = ref_tokens * strength
        
        if k.dim() == 4 and ref_kv.dim() == 3:
            batch_size, num_heads, seq_len, head_dim = k.shape
            ref_kv = ref_kv.view(1, -1, num_heads, head_dim)
            ref_kv = ref_kv.permute(0, 2, 1, 3)
            ref_kv = ref_kv.expand(batch_size, -1, -1, -1)
            cat_dim = 2
        elif k.dim() == 3 and ref_kv.dim() == 3:
            batch_size = k.shape[0]
            ref_kv = ref_kv.expand(batch_size, -1, -1)
            cat_dim = 1
        else:
            if ref_kv.shape[0] != k.shape[0]:
                expand_shape = [k.shape[0]] + [-1] * (ref_kv.dim() - 1)
                ref_kv = ref_kv.expand(*expand_shape)
            cat_dim = 2 if k.dim() == 4 else 1

        k_combined = torch.cat([k, ref_kv], dim=cat_dim)
        v_combined = torch.cat([v, ref_kv], dim=cat_dim)
        
        return {
            "q": q,
            "k": k_combined,
            "v": v_combined,
            "pe": kwargs.get("pe", None),
            "attn_mask": kwargs.get("attn_mask", None)
        }

    return attn1_patch_fn


def apply_fmars_patches_to_model(model, fmars_data):
    """
    FMARS verisini alır ve ComfyUI MODEL objesine patch'ler.
    """
    if model is None or fmars_data is None:
        return model
        
    reference_tokens = fmars_data.get("reference_tokens")
    strength = fmars_data.get("strength", 1.0)
    
    if reference_tokens is None:
        return model
        
    patched_model = model.clone()
    
    # Cross-Attention Patch
    cross_patch = create_fmars_cross_attention_patch(reference_tokens, strength)
    if cross_patch is not None:
        patched_model.set_model_attn2_patch(cross_patch)
    
    # Self-Attention Patch (aktif!)
    self_patch = create_fmars_self_attention_patch(reference_tokens, strength * 0.5)
    if self_patch is not None:
        patched_model.set_model_attn1_patch(self_patch)
        
    return patched_model