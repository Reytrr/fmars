import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceTokenProjector(nn.Module):
    """
    VAE latent'lerini transformer'ın anlayacağı token formatına dönüştürür.
    
    VAE latent shape: (batch, channels, h, w) → örn: (1, 16, 96, 96)
    Hedef: (batch, num_tokens, transformer_dim)
    
    FLUX.2 transformer dim = 15360 (değişti!)
    """
    
    def __init__(self, in_channels=16, transformer_dim=15360, max_tokens=4096):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.max_tokens = max_tokens
        
        # 16 kanal latent → transformer_dim projeksiyon
        self.proj = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.SiLU(),
            nn.Linear(1024, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )
        
        # Token sayısı çok büyükse downsample
        self.token_compress = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim),
            nn.SiLU(),
            nn.Linear(transformer_dim, transformer_dim),
        )
    
    def forward(self, latent):
        """
        Args:
            latent: (batch, channels, h, w) VAE latent
        Returns:
            tokens: (batch, num_tokens, transformer_dim)
        """
        b, c, h, w = latent.shape
        
        # (b, c, h, w) → (b, h*w, c)
        x = latent.flatten(2).transpose(1, 2)
        
        # Projeksiyon: (b, h*w, c) → (b, h*w, transformer_dim)
        tokens = self.proj(x)
        
        # Token sayısı çok büyükse sıkıştır
        num_tokens = tokens.shape[1]
        if num_tokens > self.max_tokens:
            # Adaptive average pooling ile token sayısını azalt
            tokens = tokens.transpose(1, 2)  # (b, dim, num_tokens)
            tokens = F.adaptive_avg_pool1d(tokens, self.max_tokens)
            tokens = tokens.transpose(1, 2)  # (b, max_tokens, dim)
        
        # Final projeksiyon
        tokens = self.token_compress(tokens)
        
        return tokens
    
    def to_device(self, device, dtype=torch.float32):
        """Modeli belirtilen cihaza taşır"""
        self.to(device=device, dtype=dtype)
        return self


class ReferenceEncoder:
    """
    Referans görselleri işleyip FMARS attention patch'i için hazırlar.
    """
    
    def __init__(self, transformer_dim=15360, max_tokens_per_ref=2048):
        self.transformer_dim = transformer_dim
        self.max_tokens_per_ref = max_tokens_per_ref
        self._projector = None
    
    def _get_projector(self, device, dtype, in_channels=16):
        """Projector'ı lazy olarak oluşturur"""
        if self._projector is None:
            self._projector = ReferenceTokenProjector(
                in_channels=in_channels,
                transformer_dim=self.transformer_dim,
                max_tokens=self.max_tokens_per_ref
            )
            self._projector = self._projector.to_device(device, dtype)
        return self._projector
    
    def encode_references(self, reference_latents, entity_infos, device, dtype=torch.float32):
        """
        Referans latent'lerini token formatına çevirir.
        
        Args:
            reference_latents: list of (1, 16, h, w) tensors
            entity_infos: list of dict with entity metadata
            device: torch device
            dtype: torch dtype
            
        Returns:
            dict with:
                - "tokens": (batch, total_tokens, dim) - tüm referans token'ları
                - "entity_mask": (batch, total_tokens) - her token'ın hangi entity'ye ait olduğu
                - "entity_count": int
        """
        if not reference_latents:
            return None
        
        projector = self._get_projector(
            device, 
            dtype, 
            in_channels=reference_latents[0].shape[1]
        )
        
        all_tokens = []
        entity_masks = []
        
        for idx, (latent, info) in enumerate(zip(reference_latents, entity_infos)):
            # Latent'i token formatına çevir
            latent_device = latent.to(device=device, dtype=dtype)
            tokens = projector(latent_device)  # (1, num_tokens, dim)
            
            all_tokens.append(tokens)
            
            # Entity mask oluştur (her token için entity index)
            entity_mask = torch.full(
                (tokens.shape[0], tokens.shape[1]), 
                idx, 
                device=device, 
                dtype=torch.long
            )
            entity_masks.append(entity_mask)
        
        # Tüm token'ları birleştir
        combined_tokens = torch.cat(all_tokens, dim=1)  # (batch, total_tokens, dim)
        combined_mask = torch.cat(entity_masks, dim=1)   # (batch, total_tokens)
        
        return {
            "tokens": combined_tokens,
            "entity_mask": combined_mask,
            "entity_count": len(reference_latents),
            "entity_infos": entity_infos,
        }


# Global encoder instance
_encoder = None

def get_reference_encoder(transformer_dim=15360):
    """Singleton reference encoder"""
    global _encoder
    if _encoder is None:
        _encoder = ReferenceEncoder(transformer_dim=transformer_dim)
    return _encoder