import os
import torch
import numpy as np
from PIL import Image
import math

from .core.metadata_loader import MetadataLoader
from .core.image_loader import load_image_tensor
from .core.cache import VAECache
from .matcher.term_matcher import find_all_matches
from .flux.reference_encoder import get_reference_encoder
from .flux.attention_patch import apply_fmars_patches_to_model

class FMARSReferenceLoader:
    FLUX_KNOWN_ITEMS = {
        "ak47", "m4_carbine", "beretta_92",
        "m1_abrams", "t72",
        "ch47_chinook", "uh60_blackhawk",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "references_root": ("STRING", {"default": "references"}),
                "reference_resolution": ("INT", {
                    "default": 768, "min": 256, "max": 2048, "step": 16,
                }),
                "max_references": ("INT", {"default": 3, "min": 1, "max": 10}),
                "reference_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1,
                }),
                "clear_cache": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "IMAGE", "STRING")
    RETURN_NAMES = ("model", "conditioning", "preview_images", "debug_log")
    FUNCTION = "load_reference"
    CATEGORY = "FMARS"

    def load_reference(self, prompt, references_root, reference_resolution,
                       max_references, reference_strength, clear_cache,
                       model=None, clip=None, vae=None):
        
        debug_lines = []
        
        if clear_cache:
            VAECache.clear()
            debug_lines.append("🗑️ VAE Cache temizlendi.")
            
        if not references_root or references_root.strip() == "":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            references_root = os.path.join(current_dir, "references")
        elif not os.path.isabs(references_root):
            base = os.path.dirname(os.path.abspath(__file__))
            references_root = os.path.join(base, references_root)
            
        entities = MetadataLoader.load(references_root)
        if not entities:
            debug_lines.append("❌ Hiç metadata bulunamadı.")
            return self._get_empty_returns(model, clip, reference_resolution, debug_lines)
            
        all_matches = find_all_matches(prompt, entities, max_matches=max_references)
        if not all_matches:
            debug_lines.append("⚠️ Prompt'ta eşleşen askeri terim yok.")
            return self._get_empty_returns(model, clip, reference_resolution, debug_lines)
            
        matched_names = [m.get("entity", "Bilinmeyen") for m in all_matches]
        debug_lines.append(f"✅ {len(all_matches)} eşleşen entity bulundu: {', '.join(matched_names)}")
        
        all_img_paths = []
        entity_mapping = []
        
        for best in all_matches:
            entity_name = best.get("entity", "Bilinmeyen")
            if entity_name in self.FLUX_KNOWN_ITEMS:
                debug_lines.append(f"  ⏭️ {entity_name} → FLUX modelinde zaten var, referans atlandı")
                continue
                
            img_paths = []
            for p in best.get("images", []):
                if os.path.exists(p): img_paths.append(p)
            ref = best.get("reference")
            if ref and os.path.exists(ref) and ref not in img_paths:
                img_paths.append(ref)
                
            if img_paths:
                for img_path in img_paths:
                    all_img_paths.append(img_path)
                    entity_mapping.append({
                        "entity": entity_name,
                        "display_name": best.get("display_name", ""),
                        "type": best.get("type", ""),
                        "priority": best.get("priority", 0),
                        "path": img_path
                    })
            else:
                debug_lines.append(f"      ⚠️ {entity_name} için görsel dosyası bulunamadı")
                    
        if not all_img_paths or vae is None:
            if not all_img_paths:
                debug_lines.append("⚠️ Yüklenecek referans görseli kalmadı.")
            return self._get_empty_returns(model, clip, reference_resolution, debug_lines)
            
        debug_lines.append(f"📊 Toplam {len(all_img_paths)} görsel işlenecek")
        preview_tensors = []
        reference_latents = []
        valid_entity_mapping = []
        
        encoder = get_reference_encoder()
        resample_method = getattr(Image, 'Resampling', Image).LANCZOS
        
        for idx, (img_path, entity_info) in enumerate(zip(all_img_paths, entity_mapping)):
            try:
                cached_latent = VAECache.get(img_path, reference_resolution, vae)
                if cached_latent is not None:
                    latent = cached_latent
                    debug_lines.append(f"  ⚡ [{idx}] {entity_info['entity']} → Cache'den alındı")
                else:
                    img_tensor = load_image_tensor(img_path)
                    if img_tensor is None: continue
                    
                    pil_img = Image.fromarray((img_tensor.squeeze(0).numpy() * 255).astype(np.uint8))
                    pil_img = pil_img.resize((reference_resolution, reference_resolution), resample_method)
                    img_resized = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)[None, ...]
                    
                    latent = vae.encode(img_resized)
                    VAECache.set(img_path, reference_resolution, vae, latent)
                    debug_lines.append(f"  ✅ [{idx}] {entity_info['entity']} → Encode edildi")
                    
                    preview_tensors.append(img_resized) 
                    
                reference_latents.append(latent)
                valid_entity_mapping.append(entity_info)
                    
            except Exception as e:
                debug_lines.append(f"  ❌ [{idx}] Hata: {e}")
                continue
                
        if not reference_latents:
            debug_lines.append("⚠️ Hiçbir referans latent'i oluşturulamadı.")
            return self._get_empty_returns(model, clip, reference_resolution, debug_lines)
            
        encoded_data = encoder.encode_references(
            reference_latents, 
            valid_entity_mapping, 
            reference_latents[0].device, 
            reference_latents[0].dtype
        )
        combined_tokens = encoded_data["tokens"]
        entity_mask = encoded_data["entity_mask"]
        
        debug_lines.append(f"🔥 FMARS v2: {combined_tokens.shape[1]} referans token oluşturuldu")
        
        text_seq = 512
        debug_lines.append(f"📝 Text sequence length: {text_seq}")
        
        fmars_data = {
            "reference_tokens": combined_tokens,
            "strength": reference_strength,
            "entities": valid_entity_mapping,
            "entity_mask": entity_mask,
            "text_seq": text_seq,
        }
        
        out_model = model
        if model is not None:
            out_model = apply_fmars_patches_to_model(model, fmars_data)
            debug_lines.append("🚀 FMARS v2 Attention Patch'leri MODEL'e başarıyla enjekte edildi!")
        else:
            debug_lines.append("⚠️ MODEL girişi yok, patch uygulanamadı.")
            
        # ---------- Conditioning (clip.encode_from_tokens ile, return_pooled yok) ----------
        conditioning = self._create_conditioning(clip, prompt, debug_lines)
        
        # ---------- Reference latents'i conditioning içine ekle (opsiyonel) ----------
        if conditioning and len(conditioning) > 0:
            cond_dict = conditioning[0][1]
            if isinstance(cond_dict, dict):
                cond_dict["reference_latents"] = reference_latents
                cond_dict["reference_tokens"] = combined_tokens
            
        if preview_tensors:
            preview_image_output = torch.cat(preview_tensors, dim=0)
        else:
            preview_image_output = torch.zeros((1, reference_resolution, reference_resolution, 3))
            
        return (out_model, conditioning, preview_image_output, "\n".join(debug_lines))

    def _get_empty_returns(self, model, clip, reference_resolution, debug_lines):
        out_model = model if model is not None else None
        conditioning = self._create_conditioning(clip, "", debug_lines)
        preview = torch.zeros((1, reference_resolution, reference_resolution, 3))
        return (out_model, conditioning, preview, "\n".join(debug_lines))

    # ---------- Conditioning oluşturma (return_pooled olmadan) ----------
    def _create_conditioning(self, clip, prompt, debug_lines):
        TARGET_COND_DIM = 15360
        TARGET_POOLED_DIM = 6144

        if clip is None:
            debug_lines.append("⚠️ CLIP yok, manuel sıfır conditioning oluşturuluyor.")
            return self._manual_conditioning(debug_lines)

        try:
            tokens = clip.tokenize(prompt)
            
            # return_pooled argümanı olmadan encode et
            cond = clip.encode_from_tokens(tokens)  # return_pooled yok!
            
            if cond is None:
                debug_lines.append("⚠️ CLIP encode None döndü, manuel sıfır conditioning kullanılıyor.")
                return self._manual_conditioning(debug_lines)

            device = cond.device
            dtype = cond.dtype
            cond_dim = cond.shape[-1]

            # Pooled çıktıyı manuel sıfır oluştur
            pooled = torch.zeros(cond.shape[0], TARGET_POOLED_DIM, dtype=dtype, device=device)

            debug_lines.append(f"📥 CLIP çıktısı: cond {cond.shape}, pooled {pooled.shape} (manuel sıfır)")

            # COND boyutunu yükselt (tekrarlama)
            if cond_dim != TARGET_COND_DIM:
                if TARGET_COND_DIM % cond_dim == 0:
                    repeat_factor = TARGET_COND_DIM // cond_dim
                    cond = cond.repeat(1, 1, repeat_factor)
                    debug_lines.append(f"🔄 COND {cond_dim} -> {TARGET_COND_DIM} (x{repeat_factor})")
                else:
                    debug_lines.append(f"⚠️ COND boyutu {cond_dim} -> {TARGET_COND_DIM} (linear projeksiyon)")
                    import torch.nn as nn
                    proj = nn.Linear(cond_dim, TARGET_COND_DIM, bias=False).to(device, dtype=dtype)
                    cond = proj(cond)

            # POOLED boyutu zaten hedefte
            debug_lines.append("✅ Conditioning oluşturuldu (pooled sıfır).")
            return [[cond, {"pooled_output": pooled}]]

        except Exception as e:
            debug_lines.append(f"⚠️ CLIP tokenization hatası: {e}, manuel sıfır conditioning kullanılıyor.")
            return self._manual_conditioning(debug_lines)

    def _manual_conditioning(self, debug_lines):
        batch = 1
        seq_len = 512
        cond_dim = 15360
        pooled_dim = 6144
        cond = torch.zeros(batch, seq_len, cond_dim, dtype=torch.float16)
        pooled = torch.zeros(batch, pooled_dim, dtype=torch.float16)
        debug_lines.append(f"🔧 Manuel sıfır conditioning: cond {cond.shape}, pooled {pooled.shape}")
        return [[cond, {"pooled_output": pooled}]]