import timm
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from app.utils import device

print("device:", device)

# モデル
model = timm.create_model(
    "vit_tiny_patch16_224",
    pretrained=True,
    num_classes=10
)

model = model.to(device)

# transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# dataset
# train_dataset[0] => (image, label)
# for image, label in train_dataset で回せる
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# DataLoader: datasetから「学習しやすい形」でデータを取り出す仕組み
# 学習ではバッチ単位でデータを処理することが多いので、必要
# shuffle=True: データをシャッフルする
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# training loop
# dataset全体を3回繰り返す
for epoch in range(3):
    # model.train(): モデルを訓練モードにする
    model.train()

    total_loss = 0

    # images: [32, 3, 224, 224]
    # labels: [32]
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # images: [32, 3, 224, 224]
        # logits: [32, 10]
        logits = model(images)

        # loss: どれだけ間違えたかを計算
        loss = criterion(logits, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"epoch {epoch + 1} loss: {avg_loss:.4f}")

# 保存
torch.save(
    model.state_dict(),
    "cifar10_vit_tiny.pt"
)

print("saved!")