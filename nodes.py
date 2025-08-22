import torch
import torchvision.transforms.v2.functional as F

import timm

import comfy.model_management


class TimmBackboneLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "timm/vit_huge_patch14_clip_224.laion2b",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "load"

    CATEGORY = "loaders"

    def load(self, model_name: str):
        model = timm.create_model(model_name, pretrained=True)
        model.reset_classifier(0)  # remove classifier head

        model = model.to(
            device=comfy.model_management.get_torch_device(),
            dtype=torch.float16,
        )

        return (model,)


class TimmBackboneImageEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "feature_type": (
                    ["pooler_output", "hidden_state"],
                    {"default": "pooler_output"},
                ),
                "hidden_state_index": (
                    "INT",
                    {
                        "default": -1,
                        "step": 1,
                        "min": -128,
                        "max": 128,
                        "tooltip": "Index of the hidden state to extract. Only used when feature_type is 'hidden_state'. -1 means the last, -2 means the penultimate.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "encode"
    CATEGORY = "image"

    def encode(
        self,
        model,
        image: torch.Tensor,  # [b, h, w, c]
        feature_type: str,
        hidden_state_index: int,
    ):
        pixel_values = image.permute(0, 3, 1, 2)  # -> [b, c, h, w]
        pixel_values = pixel_values.to(
            device=comfy.model_management.get_torch_device(),
            dtype=torch.float16,
        )

        if feature_type == "hidden_state":
            _output, intermediates = model.forward_intermediates(
                pixel_values,
            )
            hidden_state = intermediates[hidden_state_index]
            # maybe [batch_size, num_features, N, N]
            batch_size, dim, _h, _w = hidden_state.size()

            hidden_state = hidden_state.permute(0, 2, 3, 1).reshape(batch_size, -1, dim)
            return (hidden_state,)

        elif feature_type == "pooler_output":
            last_hidden_state = model.forward_features(pixel_values)
            pooler_output = model.forward_head(last_hidden_state)
            return (pooler_output,)

        else:
            raise ValueError(f"Unknown feature type: {feature_type}")


class ImageNormalize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mean": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "0.5, 0.5, 0.5",
                    },
                ),
                "std": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "0.5, 0.5, 0.5",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalize"

    CATEGORY = "image"

    def str_to_float_list(self, s: str):
        return [float(x) for x in s.split(",")]

    def normalize(
        self,
        image: torch.Tensor,  # [b, h, w, c]
        mean: str,
        std: str,
    ):
        image = image.permute(0, 3, 1, 2)  # -> [b, c, h, w]
        image = F.normalize(
            image,
            mean=self.str_to_float_list(mean),
            std=self.str_to_float_list(std),
        ).permute(0, 2, 3, 1)  # -> [b, h, w, c]
        return (image,)


class TimmEmbedsPrint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "embeds": ("TENSOR",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "print_embeds"
    CATEGORY = "utils"

    def print_embeds(self, embeds: torch.Tensor):
        print(f"{embeds.shape}: {embeds}")
        return ()


class RGB2BGR:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"

    CATEGORY = "image"

    def convert(
        self,
        image: torch.Tensor,  # [b, h, w, c]
    ):
        if image.shape[-1] != 3:
            raise ValueError("Input image must have 3 channels (RGB).")

        # Convert RGB to BGR by reversing the last dimension
        image = torch.flip(image, dims=[-1])

        return (image,)
