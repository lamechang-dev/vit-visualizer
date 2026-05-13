from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import timm
import torch
import torch.nn.functional as F
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
    # モデルが要求するサイズにリサイズする
    transforms.Resize((224, 224)),
    # (H, W, C) → (C, H, W) に変換
    # つまり [224][224][3] → [3, 224, 224]
    # 画像をテンソルに変換する
    transforms.ToTensor(),
])

def get_embedding(image: Image.Image) -> torch.Tensor:
    # [3, 224, 224] → [1, 3, 224, 224]
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        # Encoder出力
        features = model.forward_features(x)

    # CLS token取り出し
    cls = features[:, 0]

    return cls

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # height × width × channelの構造を持つ画像を読み込む
    # channel 0 → R（赤の強さ）: 0〜255
    # channel 1 → G（緑の強さ）: 0〜255
    # channel 2 → B（青の強さ）: 0〜255
    # [height][width][channel] = [224][224][3]
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    print(image)
    # <PIL.Image.Image image mode=RGB size=844x563 at 0x10E50F4D0>

    # [3, 224, 224] → [1, 3, 224, 224] に変換
    # ViTはバッチ(複数枚)で入力を受け取る設計
    # 1枚だけ渡す場合でも「1枚のバッチ」として包んであげる必要あり
    x = transform(image).unsqueeze(0)

    # torch.Size([1, 3, 224, 224])
    print(x.shape)

    # y => logits (logitsはまだ確率ではない)
    # 合計1じゃない
    # マイナスもある
    # 何でもあり
    # logits =>
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
        # patch embedding => transformer encoder => CLS token => Linear head => logits
        y = model(x)


    # head直前で止める。Encoderの出力のこと
    # Encoderの中で Self-Attention & MLP & LayerNormが繰り替えされる
    # このfeaturesからCLSを取り出して、MLP Headで分類する
    # [1, 197, 768]
    # 197 => 197個のパッチ(196 + 1(CLS token))
    features = model.forward_features(x)
    print("features.shape:", features.shape)
    cls = features[0, 0]
    print("cls.shape:", cls.shape)
    partial_cls = cls.tolist()[:10]
    print(partial_cls)

    # print(y.shape)
    # torch.Size([1, 1000])
    # print(y)
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

@app.post("/similarity")
async def similarity(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...)
):
    contents1 = await file1.read()
    contents2 = await file2.read()

    image1 = Image.open(io.BytesIO(contents1)).convert("RGB")
    image2 = Image.open(io.BytesIO(contents2)).convert("RGB")

    emb1 = get_embedding(image1)
    emb2 = get_embedding(image2)

    similarity = F.cosine_similarity(emb1, emb2)

    return {
        "similarity": similarity.item()
    }