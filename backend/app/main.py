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

# Linear(in_features=768, out_features=1000, bias=True)
print(model.head)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # (H, W, C) → (C, H, W) に変換
    # つまり [224][224][3] → [3, 224, 224]
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # height × width × channelの構造を持つ画像を読み込む
    # [height][width][channel] = [224][224][3]
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    print(image)
    # <PIL.Image.Image image mode=RGB size=844x563 at 0x10E50F4D0>

    # [3, 224, 224] → [1, 3, 224, 224] に変換
    x = transform(image).unsqueeze(0)

    # torch.Size([1, 3, 224, 224])
    print(x.shape)

    # y = logits logitsはまだ確率ではない
    # 合計1じゃない
    # マイナスもある
    # 何でもあり
    # logts
        # dog: 12.3
        # cat: 2.1
        # car: -4.5
    # logitsを確率に変換する
        # softmaxを使う
        # softmax(logits)
        # dog: 0.999
        # cat: 0.001
        # car: 0.000
    with torch.no_grad():
        y = model(x)


    print(y.shape)
    # torch.Size([1, 1000])
    print(y)
    # tensor([[ 4.0559e-02,  1.1820e-01,  6.7152e-01,  2.2492e-01,  2.7555e-01,...
    
    # 一番logitsが大きいものを選ぶ
    # dog: 12.3
    # cat: 2.1
    # car: -4.5
    # なのでdogを選ぶ
    pred = y.argmax().item()

    return {
        "prediction": pred
    }