from .nodes import TimmBackboneLoader, TimmBackboneImageEncode, TimmEmbedsPrint

NODE_CLASS_MAPPINGS = {
    "TimmBackboneLoader": TimmBackboneLoader,
    "TimmBackboneImageEncode": TimmBackboneImageEncode,
    "TimmEmbedsPrint": TimmEmbedsPrint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TimmBackboneLoader": "Timm Backbone Loader",
    "TimmBackboneImageEncode": "Timm Backbone Image Encode",
    "TimmEmbedsPrint": "Timm Embeds Print",
}
