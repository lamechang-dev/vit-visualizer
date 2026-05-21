from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.datasets as datasets
import open_clip
import base64
import os
import cv2
from torchvision.models import ResNet18_Weights

IMAGENET_LABELS = (
    ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ViT モデル ---

# patch16なので、16x16のパッチに分割
# 224画像なら、224/16 = 14x14のパッチに分割
# 14 x 14 = 196 patch
# 各パッチは16 x 16 x 3 = 768のベクトルになる
# 1000クラス分類のモデル。ImageNetの1000クラス分類モデル。
# pretrained=True: ImageNetの重みを使用する
# pretrained=Falseだった場合は、ランダムに初期化されたモデルが作成される。ImageNet関連の知識はなし
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True
)

for block in model.blocks:
    block.attn.fused_attn = False

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
    # [3, 224, 224] → [1, 3, 224, 224] に変換
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        # Encoder出力
        # [1, 197, 768]
        features = model.forward_features(x)

    # CLS token取り出し
    # [1, 197, 768] => [1, 768]
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
_CACHE_PATH = os.path.join(_DATA_DIR, "cifar10_clip_embeddings.pt")
_NUM_IMAGES = 500  # テストセット先頭500枚を使用

# 画像 + ラベルのセット
# 検索のたびにCLIP推論を行うのは非効率なので、事前に埋め込みを計算しておく
# CIFAR-10は画像と10カテゴリ分類のデータセット

cifar10 = datasets.CIFAR10(root=_DATA_DIR, train=False, download=True)

(image, label) = cifar10[0]
print(cifar10[0])

# input: [3, 224, 224]
# output: [1, 512]
def get_clip_embedding(image: Image.Image) -> torch.Tensor:
    x = clip_preprocess(image).unsqueeze(0)

    with torch.no_grad():
        emb = clip_model.encode_image(x)

        # cosine similarity用に正規化
        emb = F.normalize(emb, dim=-1)

    return emb

# 画像をembeddingに変換する
def _load_or_compute_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    os.makedirs(_DATA_DIR, exist_ok=True)

    if os.path.exists(_CACHE_PATH):
        data = torch.load(_CACHE_PATH, weights_only=True)
        print(f"CLIP埋め込みをキャッシュから読み込みました ({len(data['labels'])}枚)")
        return data["embeddings"], data["labels"]

    print(f"CIFAR-10の{_NUM_IMAGES}枚についてCLIP埋め込みを計算中...")
    all_embs = []
    batch_size = 32

    for i in range(0, _NUM_IMAGES, batch_size):
        # 1バッチ分の画像をCLIPの前処理にかけてスタック(cifar10[j][0]は画像)
        batch = [clip_preprocess(cifar10[j][0]) for j in range(i, min(i + batch_size, _NUM_IMAGES))]
        batch_tensor = torch.stack(batch)
        with torch.no_grad():
            # 画像を意味ベクトル(512次元)に変換
            # 32枚の画像 => 512次元のベクトル(32, 512)
            embs = clip_model.encode_image(batch_tensor)
            # コサイン類似度のためにL2正規化
            embs = F.normalize(embs, dim=-1)

        all_embs.append(embs)
        print(f"  {min(i + batch_size, _NUM_IMAGES)}/{_NUM_IMAGES}枚完了")

    embeddings = torch.cat(all_embs, dim=0)  # shape: (500, 512)
    labels = torch.tensor([cifar10[i][1] for i in range(_NUM_IMAGES)])

    torch.save({"embeddings": embeddings, "labels": labels}, _CACHE_PATH)
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
