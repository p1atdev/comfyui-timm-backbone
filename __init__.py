from .nodes import (
    TimmBackboneLoader,
    TimmBackboneImageEncode,
    TimmEmbedsPrint,
    ImageNormalize,
    RGB2BGR,
)

NODE_CLASS_MAPPINGS = {
    "TimmBackboneLoader": TimmBackboneLoader,
    "TimmBackboneImageEncode": TimmBackboneImageEncode,
    "TimmEmbedsPrint": TimmEmbedsPrint,
    "TimmBackboneImageNormalize": ImageNormalize,
    "TimmBackboneRGB2BGR": RGB2BGR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimmBackboneLoader": "Timm Backbone Loader",
    "TimmBackboneImageEncode": "Timm Backbone Image Encode",
    "TimmEmbedsPrint": "Timm Embeds Print",
    "TimmBackboneImageNormalize": "Image Normalize",
    "TimmBackboneRGB2BGR": "RGB to BGR",
}
