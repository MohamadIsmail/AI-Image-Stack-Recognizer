import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from nncf import Dataset as NNCFDataset
from nncf.quantization.quantize_model import quantize

# ─────── Config ───────
CALIB_DIR = "quant_calib"
BATCH_SIZE = 8

class CalibrationDataset(Dataset):
    def __init__(self, folder):
        self.paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".png")
        ]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        tensor = self.transform(img)
        return tensor  # ✅ NO metadata


# ─────── Load Data ───────
torch_dataset = CalibrationDataset(CALIB_DIR)
torch_loader = DataLoader(torch_dataset, batch_size=BATCH_SIZE)

# NNCF will automatically handle batches, tuples, and image transforms
nncf_dataset = NNCFDataset(torch_loader)

# ─────── Load Model ───────
model = models.resnet18(pretrained=True).eval()

# Quantize using NNCF PTQ (v2.17+)
quantized_model = quantize(
    model=model,
    calibration_dataset=nncf_dataset
)

# ─────── Export to ONNX ───────
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    quantized_model,
    dummy_input,
    "resnet18_int8.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("✅ Quantized model saved to resnet18_int8.onnx")

