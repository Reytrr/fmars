import os
import torch
from collections import OrderedDict

class VAECache:
    """
    VAE Encode LRU Cache Sistemi (CPU-Optimized, Max 10 Görsel)
    
    - Maksimum 10 latent cache'ler
    - 10'a ulaştığında en eski (en az kullanılan) otomatik silinir
    - Tensor'lar CPU'ya taşınarak VRAM şişmesi önlenir
    """
    _cache = OrderedDict()
    MAX_CACHE_SIZE = 10  # 🔥 Maksimum 10 görsel cache'lenir

    @classmethod
    def _get_cache_key(cls, image_path, resolution, vae):
        abs_path = os.path.abspath(image_path)
        return f"{abs_path}_{resolution}_vae"

    @classmethod
    def get(cls, image_path, resolution, vae):
        cache_key = cls._get_cache_key(image_path, resolution, vae)
        if cache_key in cls._cache:
            cls._cache.move_to_end(cache_key)  # En son kullanılan olarak işaretle
            return cls._cache[cache_key]
        return None

    @classmethod
    def set(cls, image_path, resolution, vae, latent):
        cache_key = cls._get_cache_key(image_path, resolution, vae)
        
        # 🔥 Tensor'ı CPU'ya taşı (VRAM koruması)
        if hasattr(latent, 'cpu'):
            cached_latent = latent.detach().cpu().clone()
        else:
            cached_latent = latent
            
        cls._cache[cache_key] = cached_latent
        cls._cache.move_to_end(cache_key)
        
        # 🔥 LRU: 10'a ulaştığında en eski (en az kullanılan) öğeyi sil
        while len(cls._cache) > cls.MAX_CACHE_SIZE:
            old_key, old_tensor = cls._cache.popitem(last=False)
            del old_tensor  # Bellekten tamamen temizle

    @classmethod
    def clear(cls):
        """Cache'i manuel olarak temizler ve GPU/CPU belleğini serbest bırakır"""
        # Tüm tensor referanslarını sil
        for key in list(cls._cache.keys()):
            del cls._cache[key]
        cls._cache.clear()
        
        # 🧹 CUDA cache'ini de temizle (VRAM'i tam boşalt)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        # 🧹 Python garbage collector'ı tetikle
        try:
            import gc
            gc.collect()
        except Exception:
            pass
    
    @classmethod
    def get_stats(cls):
        """Cache istatistikleri"""
        total_bytes = 0
        for tensor in cls._cache.values():
            if hasattr(tensor, 'element_size') and hasattr(tensor, 'nelement'):
                total_bytes += tensor.element_size() * tensor.nelement()
        return {
            "items": len(cls._cache),
            "max_size": cls.MAX_CACHE_SIZE,
            "ram_usage_mb": round(total_bytes / (1024 * 1024), 2)
        }