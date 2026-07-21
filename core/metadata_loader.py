import os
import json

class MetadataLoader:
    _cache = {}

    @classmethod
    def load(cls, root_path):
        root_path = os.path.abspath(root_path)
        if root_path in cls._cache:
            return cls._cache[root_path]

        entities = []
        if not os.path.isdir(root_path):
            print(f"[FMARS] Uyarı: '{root_path}' klasörü yok.")
            cls._cache[root_path] = entities
            return entities

        military_types = {
            "weapon", "aircraft", "ship", "tank", "vehicle", "missile",
            "submarine", "helicopter", "drone", "artillery", "armored vehicle",
            "battleship", "fighter", "bomber", "carrier", "cruiser",
            "destroyer", "frigate", "corvette", "howitzer", "rocket"
        }

        for dirpath, _, files in os.walk(root_path):
            for fname in files:
                if fname.lower() == "metadata.json":
                    json_path = os.path.join(dirpath, fname)
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        etype = data.get("type", "").lower()
                        if etype in military_types:
                            if "images" in data:
                                cleaned_images = []
                                for p in data["images"]:
                                    normalized = p.replace("\\", "/")
                                    if normalized.startswith("references/"):
                                        normalized = normalized[len("references/"):]
                                    cleaned_images.append(os.path.join(root_path, normalized))
                                data["images"] = cleaned_images
                            
                            if "reference" in data:
                                ref = data["reference"].replace("\\", "/")
                                if ref.startswith("references/"):
                                    ref = ref[len("references/"):]
                                data["reference"] = os.path.join(root_path, ref)
                            
                            data["aliases"] = [a.lower() for a in data.get("aliases", [])]
                            data["display_name"] = data.get("display_name", "").lower()
                            entities.append(data)
                    except Exception as e:
                        print(f"[FMARS] {json_path} okunamadı: {e}")

        cls._cache[root_path] = entities
        return entities