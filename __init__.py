from .nodes import FMARSReferenceLoader

NODE_CLASS_MAPPINGS = {
    "FMARS Reference Loader": FMARSReferenceLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FMARS Reference Loader": "FMARS Reference Loader (Military)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']