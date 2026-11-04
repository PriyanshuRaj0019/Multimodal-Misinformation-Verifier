
import torch
import transformers

from PIL import Image
from transformers import CLIPModel, CLIPProcessor


print("=" * 60)
print("MMV — CLIP ENVIRONMENT CHECK")
print("=" * 60)

print("\nPackage versions:")
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

print("\nCLIP imports:")
print("CLIPModel:", CLIPModel.__name__)
print("CLIPProcessor:", CLIPProcessor.__name__)

print("\nImage library:")
print("Pillow:", Image.__version__ if hasattr(Image, "__version__") else "Available")

print("\nGPU availability:")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    properties = torch.cuda.get_device_properties(0)

    print("GPU:", properties.name)
    print(
        "GPU memory:",
        f"{properties.total_memory / (1024 ** 3):.2f} GB",
    )
    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )

    print(
        "CUDA version used by PyTorch:",
        torch.version.cuda,
    )
else:
    print("WARNING: CUDA is unavailable. CPU inference would be slower.")

print("\nRESULT: CLIP imports and environment checks completed.")
