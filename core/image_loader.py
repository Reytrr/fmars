import os
import math
import torch
import numpy as np
from PIL import Image

def load_image_tensor(image_path):
    if not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img)[None, ...]
    except Exception as e:
        print(f"[FMARS] Görsel yükleme hatası: {image_path} - {e}")
        return None

def create_grid(img_tensors, grid_res=384):
    """Birden fazla görseli grid halinde birleştirir (Önizleme için)"""
    if not img_tensors:
        return None
    
    n = len(img_tensors)
    if n == 1:
        return img_tensors[0]
    
    # Önizleme için maksimum 9 görsel göster
    if n > 9:
        img_tensors = img_tensors[:9]
        n = 9

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    resized_imgs = []
    resample_method = getattr(Image, 'Resampling', Image).LANCZOS
    
    for t in img_tensors:
        pil_img = Image.fromarray((t.squeeze(0).numpy() * 255).astype(np.uint8))
        pil_img = pil_img.resize((grid_res, grid_res), resample_method)
        resized_imgs.append(pil_img)
    
    grid_width = cols * grid_res
    grid_height = rows * grid_res
    grid_img = Image.new('RGB', (grid_width, grid_height), (30, 30, 30))
    
    for idx, pil_img in enumerate(resized_imgs):
        row = idx // cols
        col = idx % cols
        x = col * grid_res
        y = row * grid_res
        grid_img.paste(pil_img, (x, y))
    
    arr = np.array(grid_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def merge_images_tensors(tensors, direction='horizontal', target_res=1024):
    if not tensors:
        return None
    
    if isinstance(target_res, (tuple, list)):
        target_res = target_res[0] if target_res else 1024
    target_res = int(target_res)

    pil_imgs = []
    for t in tensors:
        arr = (t.squeeze(0).numpy() * 255).astype(np.uint8)
        pil_imgs.append(Image.fromarray(arr))

    if len(pil_imgs) == 1:
        img = pil_imgs[0]
    else:
        if direction == 'horizontal':
            widths, heights = zip(*(i.size for i in pil_imgs))
            total_width = sum(widths)
            max_height = max(heights)
            new_img = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in pil_imgs:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.width
        else:
            widths, heights = zip(*(i.size for i in pil_imgs))
            max_width = max(widths)
            total_height = sum(heights)
            new_img = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for img in pil_imgs:
                new_img.paste(img, (0, y_offset))
                y_offset += img.height
        img = new_img

    resample_method = getattr(Image, 'Resampling', Image).LANCZOS
    img_resized = img.resize((target_res, target_res), resample_method)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def load_and_merge_images(image_paths, direction='horizontal', target_res=1024):
    tensors = []
    for path in image_paths:
        t = load_image_tensor(path)
        if t is not None:
            tensors.append(t)
            
    if not tensors:
        return None
        
    return merge_images_tensors(tensors, direction, target_res)