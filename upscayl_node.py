import subprocess
import torch
from PIL import Image
import numpy as np
from torchvision.transforms.v2 import ToPILImage
import tempfile

def pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    """Converts a PIL Image or a list of PIL Images to a tensor."""

    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)

    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)

class Upscayl_Upscaler:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": ("INT", {
                    "default": 2048,
                    "min": 1024,
                    "max": 4096,
                    "step": 64,
                    "display": "number",
                }),
                "model": (["upscayl-lite-4x",
                           "upscayl-standard-4x",
                           "ultrasharp-4x",
                           "ultramix-balanced-4x",
                           "remacri-4x",
                           "digital-art-4x",
                           "high-fidelity-4x"],
                          ),
                "upscayl_path": ("STRING", {
                    "default": "/opt/Upscayl/resources/bin/upscayl-bin",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("out_image",)
    FUNCTION = "upscayl_upscale"
    CATEGORY = "upscale tools"

    def upscayl_upscale(self, image, resolution, model, upscayl_path):
        
        with torch.no_grad():
            pil_image = ToPILImage()(image.permute([0, 3, 1, 2])[0]).convert("RGB")
        input_path = tempfile.NamedTemporaryFile(suffix=".png").name
        pil_image.save(input_path)
        upscayl_path = upscayl_path
        output_path = tempfile.NamedTemporaryFile(suffix=".png").name
        
        cmd = [
            upscayl_path,
            "-i", input_path,
            "-o", output_path,
            "-n", model,
            "-w", resolution,
            "-f", "png"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("图片处理成功")
            else:
                print("错误:", result.stderr)
        except Exception as e:
            print("执行出错:", str(e))
        
        out_image = open(output_path, "rb")
        return (pil2tensor(out_image),)
            