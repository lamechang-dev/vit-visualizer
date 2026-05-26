import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.datasets as datasets
from app.model import model, transform, clip_model, clip_preprocess
from app.utils import device

_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data")
_CACHE_PATH = os.path.join(_DATA_DIR, "cifar10_clip_embeddings.pt")
_NUM_IMAGES = 500  # テストセット先頭500枚を使用

# 画像 + ラベルのセット
# 検索のたびにCLIP推論を行うのは非効率なので、事前に埋め込みを計算しておく
# CIFAR-10は画像と10カテゴリ分類のデータセット

cifar10 = datasets.CIFAR10(root=_DATA_DIR, train=False, download=True)

(image, label) = cifar10[0]
print(cifar10[0])

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

# input: [3, 224, 224]
# output: [1, 512]
def get_clip_embedding(image: Image.Image) -> torch.Tensor:
    x = clip_preprocess(image).unsqueeze(0).to(device)

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
