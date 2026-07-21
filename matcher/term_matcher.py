import re

# ============================================================
# BAĞLAM (CONTEXT) ANALİZİ İÇİN ANAHTAR KELİMELER
# ============================================================
CONTEXT_KEYWORDS = {
    "vehicle": [
        "tank", "vehicle", "armor", "armored", "tracked", "wheeled",
        "ifv", "apc", "spg", "self-propelled", "sherman", "patton",
        "abrams", "leopard", "btr", "bmp", "m113", "bradley",
        "stryker", "humvee", "jltv", "mrap", "t-72", "t-90"
    ],
    "weapon": [
        "rifle", "gun", "weapon", "carbine", "firearm", "assault",
        "machine gun", "pistol", "handgun", "smg", "submachine",
        "shotgun", "sniper", "marksman", "lmg", "hmg", "mmg",
        "gpmg", "sidearm", "magazine", "ammunition", "caliber"
    ],
    "aircraft": [
        "aircraft", "plane", "jet", "fighter", "bomber", "helicopter",
        "chopper", "uav", "drone", "aviation", "aerial", "sortie",
        "cockpit", "fuselage", "wing", "tail rotor", "nap-of-the-earth"
    ],
    "ship": [
        "ship", "boat", "naval", "navy", "vessel", "destroyer",
        "cruiser", "frigate", "submarine", "carrier", "battleship",
        "corvette", "hull", "deck", "turret", "propeller"
    ],
    "missile": [
        "missile", "rocket", "sam", "surface-to-air", "atgm",
        "anti-tank", "cruise", "launcher", "payload"
    ]
}

TYPE_NORMALIZATION = {
    "vehicle": {"vehicle", "vehicles", "tank", "armored vehicle"},
    "weapon": {"weapon", "weapons"},
    "aircraft": {"aircraft", "helicopter", "drone", "fighter", "bomber"},
    "ship": {"ship", "submarine", "battleship", "carrier", "cruiser", "destroyer", "frigate", "corvette"},
    "missile": {"missile", "rocket", "artillery"}
}

# Tek başına kullanıldığında yanlış eşleşmeye yol açan jenerik kelimeler
# (Çok kelimeli ifadeler如 "desert eagle" veya "machine gun" etkilenmez)
GENERIC_SINGLE_WORDS = {
    "desert", "hawk", "stealth", "zero", "black", "white", "red",
    "green", "blue", "type", "class", "system", "block", "motion",
    "dark", "light", "night", "day", "air", "ground", "sea",
    "gun", "rifle", "tank", "plane", "ship", "car", "truck",
    "machine", "submachine", "automatic", "semi-automatic",
    "soviet", "union", "german", "british", "american", "french",
    "russian", "chinese", "japanese", "israeli", "turkish",
    "angeles", "los", "ford", "ohio", "typhoon", "uss",
    "eagle", "mirage", "global", "super",
    "1", "2", "3", "4", "5", "i", "ii", "iii", "iv", "v"
}

# Bilinen askeri kısaltmalar (çoğul -> tekil dönüşümü için)
KNOWN_MILITARY_ABBREVS = {
    'ak', 'pkm', 'rpg', 'lmg', 'hmg', 'mmg', 'gpmg', 'smg', 'ar', 'sar',
    'saw', 'mgs', 'atgm', 'sam', 'mbt', 'apc', 'ifv', 'spaag',
    'uav', 'icbm', 'slbm', 'nco', 'pow', 'kia', 'wia', 'mia',
}


def _strip_negative_prompt(prompt):
    """
    Negatif prompt bloğunu metinden ayıkla.
    String bazlı çalışır (tek satırlık prompt'larda bile doğru çalışır).
    """
    prompt_lower = prompt.lower()
    
    # En belirgin anahtar kelimelerden başla
    negative_keywords = [
        "negative prompt block:",
        "negative prompt:",
        "negatif prompt:",
        "negative prompt block",
        "negative prompt",
    ]
    
    # İlk bulunan anahtar kelimeden öncesini al
    for kw in negative_keywords:
        idx = prompt_lower.find(kw)
        if idx >= 0:
            result = prompt[:idx].strip()
            if result:
                return result
            return ""
    
    # Fallback: Satır bazlı kontrol (basit "negative:" formatı için)
    lines = prompt.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        if line_lower.startswith('negative:') or line_lower.startswith('negatif:'):
            return '\n'.join(lines[:i]).strip()
    
    return prompt


def _normalize_prompt(prompt):
    """
    Prompt'taki çoğul silah/aracı isimlerini tekil hale getirir.
    Sadece sayı+harf kombinasyonları ve bilinen askeri kısaltmalar için çalışır.
    Genel İngilizce kelimelere DOKUNMAZ (örn: "loss" -> "los" YAPMAZ).
    """
    normalized = prompt
    
    # 1. Kural: Sayı içeren kelimeler (M70s -> M70, AK47s -> AK47)
    normalized = re.sub(r'\b([a-zA-Z]*\d+[a-zA-Z]*)s\b', r'\1', normalized, flags=re.IGNORECASE)
    
    # 2. Kural: Bilinen askeri kısaltmalar (AKs -> AK, PKMs -> PKM)
    for abbrev in KNOWN_MILITARY_ABBREVS:
        pattern = r'\b' + re.escape(abbrev) + r's\b'
        normalized = re.sub(pattern, abbrev, normalized, flags=re.IGNORECASE)
    
    return normalized


def _get_valid_terms(ent):
    """
    Geçerli eşleşme terimlerini döndür.
    Display name ve aliases listesini kullanır, çoğul hallerini de ekler.
    Tehlikeli tek kelimeleri filtreler.
    """
    terms = set()
    
    # 1. Display Name (tam hali ve çoğulu)
    display_name = ent.get("display_name", "")
    if display_name:
        dn_lower = display_name.lower().strip()
        if len(dn_lower) >= 2:
            # Tek kelimeli ve jenerik listedeyse atla
            if len(dn_lower.split()) == 1 and dn_lower in GENERIC_SINGLE_WORDS:
                pass
            else:
                terms.add(dn_lower)
                terms.add(dn_lower + "s")
    
    # 2. Aliases listesi
    for alias in ent.get("aliases", []):
        alias_lower = alias.lower().strip()
        if not alias_lower or len(alias_lower) < 2:
            continue
        
        # Tek kelimeli ve jenerik listedeyse atla
        words = alias_lower.split()
        if len(words) == 1 and alias_lower in GENERIC_SINGLE_WORDS:
            continue
        
        terms.add(alias_lower)
        terms.add(alias_lower + "s")
    
    return terms


def _get_entity_category(entity_type):
    """Entity tipini ana kategorilere (vehicle, weapon, vb.) eşleştir"""
    etype = entity_type.lower().strip()
    for category, types in TYPE_NORMALIZATION.items():
        if etype in types:
            return category
    return None


def _calculate_context_boost(prompt_lower, entity_type):
    """Prompt'taki bağlam kelimelerine göre entity'ye bonus puan verir."""
    category = _get_entity_category(entity_type)
    if not category or category not in CONTEXT_KEYWORDS:
        return 0
    
    boost = 0
    for keyword in CONTEXT_KEYWORDS[category]:
        if re.search(r'\b' + re.escape(keyword) + r'\b', prompt_lower):
            boost += 15
    return min(boost, 30)


def _calculate_frequency_boost(entity, prompt_lower):
    """Entity'nin prompt'ta kaç kez geçtiğine göre bonus puan verir."""
    count = 0
    terms = _get_valid_terms(entity)
    for term in terms:
        if len(term) >= 2:
            matches = re.findall(r'\b' + re.escape(term) + r'\b', prompt_lower)
            count += len(matches)
    return count * 15


def _score_entity(entity, prompt_lower):
    """Bir entity'nin nihai skorunu hesapla (Base Priority + Context + Frequency)"""
    base_priority = entity.get("priority", 0)
    entity_type = entity.get("type", "")
    context_boost = _calculate_context_boost(prompt_lower, entity_type)
    frequency_boost = _calculate_frequency_boost(entity, prompt_lower)
    return base_priority + context_boost + frequency_boost


def find_best_match(prompt, entities):
    """
    Prompt'ta geçen askeri terimlerden EN YÜKSEK skora sahip olanı bulur.
    Bağlam (context) ve frekans analizini kullanır.
    """
    clean_prompt = _strip_negative_prompt(prompt)
    normalized_prompt = _normalize_prompt(clean_prompt)
    prompt_lower = normalized_prompt.lower()
    
    matches = []
    for ent in entities:
        terms = _get_valid_terms(ent)
        if not terms:
            continue
        
        # Her terimi ayrı ayrı kontrol et (dev regex yerine, daha sağlam)
        for term in terms:
            if len(term) < 2:
                continue
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, prompt_lower):
                matches.append(ent)
                break
    
    if not matches:
        return None
    
    return max(matches, key=lambda x: _score_entity(x, prompt_lower))


def find_all_matches(prompt, entities, max_matches=5):
    """
    Prompt'taki TÜM eşleşen askeri terimleri bulur.
    Bağlam (context) ve frekans analizine göre skora sahip sıralı liste döndürür.
    """
    clean_prompt = _strip_negative_prompt(prompt)
    normalized_prompt = _normalize_prompt(clean_prompt)
    prompt_lower = normalized_prompt.lower()
    
    matches = []
    matched_entities = set()
    
    for ent in entities:
        terms = _get_valid_terms(ent)
        if not terms:
            continue
        
        entity_key = ent.get("entity", "")
        matched = False
        
        # Her terimi ayrı ayrı kontrol et
        for term in terms:
            if len(term) < 2:
                continue
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, prompt_lower):
                matched = True
                break
        
        if matched and entity_key not in matched_entities:
            matched_entities.add(entity_key)
            matches.append(ent)
    
    if not matches:
        return []
    
    # Bağlam ve frekans destekli skora göre azalan sırada sırala
    matches.sort(key=lambda x: _score_entity(x, prompt_lower), reverse=True)
    return matches[:max_matches]