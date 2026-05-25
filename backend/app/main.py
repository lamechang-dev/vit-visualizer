from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn.functional as F
import base64
from app.model import cifar10_model, model, transform, IMAGENET_LABELS, clip_model, clip_tokenizer, CIFAR10_CLASSES
from app.inference import get_embedding, get_clip_embedding, cifar10, _cifar10_embeddings
from app.utils import device

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    # logitsは合計1じゃない。マイナスもある。何でもあり得る。
    # logits =>
        # dog: 12.3
        # cat: 2.1
        # car: -4.5
    with torch.no_grad():
        # patch embedding => transformer encoder => CLS token => Linear head => logits
        # [1, 1000]
        logits: torch.Tensor = model(x)


    # forward_features: head直前で止める。Encoderの出力を取得する。
    # Encoderの中で Self-Attention & MLP & LayerNormが繰り替えされる
    # このfeaturesからCLSを取り出して、MLP Headで分類する
    features = model.forward_features(x)
    # [1, 197, 768]
    # 197 => 197個のパッチ(196 + 1(CLS token))
    print("features.shape:", features.shape)
    cls = features[0, 0]
    # [768]
    print("cls.shape:", cls.shape)

    # 確率のトップ10のくらすIDと確率を表示
    # softmax(): 確率分布に変換
    softmax = logits.softmax(dim=-1)
    topk = softmax[0].topk(10)
    indices = topk.indices.tolist()
    values = topk.values.tolist()
    print("logits トップ10:", indices, values)

    # 一番logitsが大きいものを選ぶ
    # dog: 12.3
    # cat: 2.1
    # car: -4.5
    # なのでdogを選ぶ
    pred = IMAGENET_LABELS[logits.argmax().item()]

    return {
        "prediction_id": logits.argmax().item(),
        "prediction": pred
    }

@app.post("/predict-cifar10")
async def predict_cifar10(
    file: UploadFile = File(...)
):
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = cifar10_model(x)

    print("logits.shape:", logits.shape)
    # [1, 10]

    probs = logits.softmax(dim=-1)

    topk = probs[0].topk(10)

    indices = topk.indices.tolist()
    values = topk.values.tolist()

    print("top10:", indices, values)

    pred_id = logits.argmax(dim=-1).item()

    pred_label = CIFAR10_CLASSES[pred_id]

    return {
        "prediction_id": pred_id,
        "prediction": pred_label,
        "probabilities": {
            CIFAR10_CLASSES[i]: float(probs[0][i])
            for i in range(10)
        }
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


@app.get("/clip-search")
async def clip_search(query: str, top_k: int = 9):
    # テキストクエリをCLIPで埋め込みに変換
    tokens = clip_tokenizer([query])
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens)
        text_emb = F.normalize(text_emb, dim=-1)  # shape: (1, 512)

    # 全画像との内積 = L2正規化済みなのでコサイン類似度と等価
    # shape: (500, 512) @ (512, 1) = (500, 1)
    # 500画像とそれぞれの検索文の類似度
    # .Tで転置
    # squeezeで長さ1の次元を消す
    scores = (_cifar10_embeddings @ text_emb.T).squeeze()

    # 類似度が高い順にソートして、上位k個を取得
    top_indices = torch.sort(scores, descending=True)[1][:top_k]

    results = []
    for idx in top_indices:
        img_pil, label = cifar10[int(idx)]
        # 32x32 → 128x128 にアップスケール（視認性向上のため）
        img_display = img_pil.resize((128, 128), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        img_display.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        results.append({
            "image": f"data:image/png;base64,{img_b64}",
            "label": CIFAR10_CLASSES[label],
            "score": float(scores[idx]),
        })

    return {"query": query, "results": results}

@app.post("/similar-images")
async def similar_images(
    file: UploadFile = File(...),
    top_k: int = 5
):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # アップロード画像のembedding
    query_emb = get_clip_embedding(image)

    # shape:
    # (500, 512) @ (512, 1)
    # => (500, 1)
    scores = (
        _cifar10_embeddings
        @ query_emb.T
    ).squeeze()

    # 類似度高い順
    top_indices = torch.argsort(scores, descending=True)[:top_k]

    results = []

    for idx in top_indices:
        img_pil, label = cifar10[int(idx)]

        # 見やすく拡大
        img_display = img_pil.resize(
            (128, 128),
            Image.Resampling.NEAREST
        )

        # base64化
        buf = io.BytesIO()
        img_display.save(buf, format="PNG")

        img_b64 = base64.b64encode(
            buf.getvalue()
        ).decode()

        results.append({
            "image": f"data:image/png;base64,{img_b64}",
            "label": CIFAR10_CLASSES[label],
            "score": float(scores[idx]),
        })

    return {
        "results": results
    }
