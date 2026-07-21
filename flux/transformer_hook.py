"""
FMARS v2 - Transformer Hook
FLUX.2 transformer'ına attention patch ekler.
"""
import torch
from .attention_patch import FMARSAttentionPatch


def apply_fmars_patch(model, reference_tokens, strength=1.0, entity_weights=None):
    """
    FLUX.2 modeline FMARS attention patch'ı ekler.
    
    Args:
        model: FLUX.2 diffusion modeli
        reference_tokens: List[torch.Tensor] - Her referans için token tensor'ları
        strength: Referans gücü (0.0 - 2.0)
        entity_infos: Her referans için metadata
    
    Returns:
        Patch uygulanmış model
    """
    if not reference_tokens:
        return model
    
    # FMARS AttentionPatch oluştur
    patch = FMARSAttentionPatch(
        reference_tokens=reference_tokens,
        strength=strength,
        entity_infos=entity_infos
    )
    
    # Model'e patch ekle
    # "attn1_patch" = self-attention (img img)
    # "attn2_patch" = cross-attention (text img)
    # FLUX.2'de reference image için "attn2_patch" (cross-attention) kullanılır
    
    # "double" = hem K hem V'ye uygula
    model.set_model_patch("attn2_patch", patch, "double")
    
    return model


def patch_model_for(model, fmars_data):
    """
    transformer_options["fmars"] içinden veriyi alıp model'e patch eder.
    
    fmars_data = {
        "reference_tokens": [...],
        "strength": 1.2,
        "entities": [...]
    }
    """
    if fmars_data is None:
        return model
    
    reference_tokens = fmars_data.get("reference_tokens", [])
    strength = fmars_data.get("strength", 1.0)
    entities = fmars_data.get("entities", [])
    
    if not reference_tokens:
        return model
    
    return apply_fmars_patch(model, reference_tokens, strength, entities)