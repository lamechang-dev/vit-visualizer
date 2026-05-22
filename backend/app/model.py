import timm
import torch
from torchvision import transforms
from torchvision.models import ResNet18_Weights
import open_clip

IMAGENET_LABELS = (
    ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
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

# --- CLIP モデル ---

# CLIP ViT-B/32 を読み込む
# CLIPモデル：テキストと画像を同じ埋め込み空間に射影するモデル
# モデルの読み込み
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
# tokenizer：テキストをトークンに変換する
# "a dog running" => ["a", "dog", "running"] => [15496, 16390, 3393](tokenID)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_model.eval()

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
