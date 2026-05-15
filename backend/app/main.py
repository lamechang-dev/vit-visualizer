from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
import open_clip
import numpy as np
import base64
import os

app = FastAPI()

# --- ViT モデル ---

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


# --- CLIP + CIFAR-10 キーワード検索 ---

# CLIP ViT-B/32 を読み込む
# CLIPモデル：テキストと画像を同じ埋め込み空間に射影するモデル
# モデルの読み込み
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
# tokenizer：テキストをトークンに変換する
# "a dog running" => ["a", "dog", "running"] => [15496, 16390, 3393](tokenID)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
_CACHE_PATH = os.path.join(_DATA_DIR, "cifar10_clip_embeddings.npz")
_NUM_IMAGES = 500  # テストセット先頭500枚を使用

# 画像 + ラベルのセット
# 検索のたびにCLIP推論を行うのは非効率なので、事前に埋め込みを計算しておく
# CIFAR-10は画像と10カテゴリ分類のデータセット

cifar10 = datasets.CIFAR10(root=_DATA_DIR, train=False, download=True)

print(cifar10[0])

# 画像をembeddingに変換する
def _load_or_compute_embeddings() -> tuple[np.ndarray, np.ndarray]:
    os.makedirs(_DATA_DIR, exist_ok=True)

    if os.path.exists(_CACHE_PATH):
        data = np.load(_CACHE_PATH)
        print(f"CLIP埋め込みをキャッシュから読み込みました ({len(data['labels'])}枚)")
        return data["embeddings"], data["labels"]

    print(f"CIFAR-10の{_NUM_IMAGES}枚についてCLIP埋め込みを計算中...")
    all_embs = []
    batch_size = 32

    for i in range(0, _NUM_IMAGES, batch_size):
        # 1バッチ分の画像をCLIPの前処理にかけてスタック
        batch = [clip_preprocess(cifar10[j][0]) for j in range(i, min(i + batch_size, _NUM_IMAGES))]
        batch_tensor = torch.stack(batch)
        with torch.no_grad():
            # 画像を意味ベクトル(512次元)に変換
            # 32枚の画像 => 512次元のベクトル(32, 512)
            embs = clip_model.encode_image(batch_tensor)
            # コサイン類似度のためにL2正規化
            embs = F.normalize(embs, dim=-1)
        all_embs.append(embs.numpy())
        print(f"  {min(i + batch_size, _NUM_IMAGES)}/{_NUM_IMAGES}枚完了")

    embeddings = np.concatenate(all_embs, axis=0)  # shape: (500, 512)
    labels = np.array([cifar10[i][1] for i in range(_NUM_IMAGES)])

    np.savez(_CACHE_PATH, embeddings=embeddings, labels=labels)
    print("CLIP埋め込みをキャッシュに保存しました")
    return embeddings, labels

_cifar10_embeddings, _cifar10_labels = _load_or_compute_embeddings()


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
    scores = (_cifar10_embeddings @ text_emb.numpy().T).squeeze()

     # 類似度が高い順にソートして、上位k個を取得
    top_indices = np.argsort(scores)[::-1][:top_k]

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
