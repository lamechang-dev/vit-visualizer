from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import timm
import torch
from torchvision import transforms

app = FastAPI()

model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        y = model(x)

    pred = y.argmax().item()

    return {
        "prediction": pred
    }