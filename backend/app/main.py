from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import torch.nn.functional as F
import base64
import numpy as np
from app.model import cifar10_model, model, transform, IMAGENET_LABELS, clip_model, clip_tokenizer, CIFAR10_CLASSES
from app.inference import get_embedding, get_clip_embedding, cifar10, _cifar10_embeddings
from app.utils import device, extract_frames
from app.model import grounding_processor, grounding_model, sam_processor, sam_model

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

    print("device:", device)

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
    x = transform(image).unsqueeze(0) # type: ignore
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
    pred = IMAGENET_LABELS[int(logits.argmax().item())]

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

    x = transform(image).unsqueeze(0).to(device) # type: ignore

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

@app.post("/multi-label")
async def multi_label(
    file: UploadFile = File(...),
    top_k: int = 5
):
    CANDIDATE_LABELS = [
        "dog",
        "cat",
        "car",
        "truck",
        "bird",
        "horse",
        "frog",
        "ship",
        "airplane",
        "grass",
        "tree",
        "person",
        "road",
        "sky",
        "building",
    ]

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # -------------------------
    # 画像embedding
    # -------------------------
    image_emb = get_clip_embedding(image)
    # shape: (1, 512)

    # -------------------------
    # テキストembedding
    # -------------------------
    tokens = clip_tokenizer(
        CANDIDATE_LABELS
    ).to(device)

    with torch.no_grad():
        text_embs = clip_model.encode_text(tokens)

    # L2 normalize
    text_embs = F.normalize(
        text_embs,
        dim=-1
    )
    # shape: (N, 512)

    # -------------------------
    # cosine similarity
    # -------------------------
    scores = (
        image_emb
        @ text_embs.T
    ).squeeze(0)

    # shape: (N)

    # -------------------------
    # top-k
    # -------------------------
    top_scores, top_indices = torch.topk(
        scores,
        k=min(top_k, len(CANDIDATE_LABELS))
    )

    results = []

    for score, idx in zip(
        top_scores,
        top_indices
    ):
        results.append({
            "label": CANDIDATE_LABELS[int(idx)],
            "score": float(score),
        })

    return {
        "predictions": results
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


@app.post("/extract-frames")
async def extract_frames_endpoint(
    file: UploadFile = File(...),
    fps: int = Form(5)
):
    contents = await file.read()
    frames = extract_frames(contents, target_fps=fps)
    return {
        "total": len(frames),
        "fps": fps,
        "frames": frames,
    }


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    labels: str = Form("dog . cat . person .")
):
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    if not labels.rstrip().endswith("."):
        labels = labels.rstrip() + " ."

    # -------------------------
    # preprocess
    # -------------------------
    # 画像: resize => normalize => to tensor
    # テキスト: tokenizer => to tensor
    # return_tensors="pt" => PyTorchのテンソルに変換して、の指定
    inputs = grounding_processor( # type: ignore
        images=image,
        text=labels,
        return_tensors="pt"
    ).to(device)

    # -------------------------
    # inference
    # -------------------------
    with torch.no_grad():
        # ViT patch embedding
        # text => Transformer token embedding
        # その後： Cross Attention
        outputs = grounding_model(**inputs) # type: ignore

    # -------------------------
    # post process
    # -------------------------
    results = (
        grounding_processor
        .post_process_grounded_object_detection( # type: ignore
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
    )

    result = results[0]

    detections = []

    for score, label, box in zip(
        result["scores"],
        result["labels"],
        result["boxes"]
    ):
        detections.append({
            "label": label,
            "score": float(score),
            "box": [
                float(v)
                for v in box.tolist()
            ]
        })

    return {
        "detections": detections
    }


@app.post("/detect_with_pixel")
async def detect_with_pixel(
    file: UploadFile = File(...),
    labels: str = Form("dog . cat . person .")
):
    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    if not labels.rstrip().endswith("."):
        labels = labels.rstrip() + " ."

    # -------------------------
    # Grounding DINO で bbox 検出
    # -------------------------
    inputs = grounding_processor( # type: ignore
        images=image,
        text=labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = grounding_model(**inputs) # type: ignore

    results = (
        grounding_processor
        .post_process_grounded_object_detection( # type: ignore
            outputs,
            inputs.input_ids,
            threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )
    )

    # [{'scores': tensor([0.7818, 0.3842], device='mps:0'), 'boxes': tensor([[175.5537,  74.6405, 196.1769,  95.0299],
    # [175.2701,  95.3338, 195.7438, 114.0614]], device='mps:0'), 'text_labels': ['duck', 'duck'], 'labels': ['duck', 'duck']}]
    # box: [x1, y1, x2, y2]
    print("results:", results)

    result = results[0]

    if len(result["boxes"]) == 0:
        return {"segments": []}

    # -------------------------
    # SAM でピクセルマスクを生成
    # -------------------------
    # SAMはbox promptを [[x1, y1, x2, y2]] の形式で受け取る
    boxes = result["boxes"].tolist()

    sam_inputs = sam_processor(
        images=image,
        input_boxes=[[boxes]],
        return_tensors="pt"
    )

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    # (1, num_boxes, 3, H, W) → best mask を選択
    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0]
    # shape: (num_boxes, 3, H, W) — 3候補のうちスコアが最高のものを選ぶ

    iou_scores = sam_outputs.iou_scores[0]
    # shape: (num_boxes, 3)

    segments = []

    image_np = np.array(image)

    # (height, width, channel(RGB))
    # (563, 844, 3)
    print("image_np.shape", image_np.shape)

    for i, (score, label, box) in enumerate(zip(
        result["scores"],
        result["labels"],
        result["boxes"]
    )):
        best_mask_idx = int(iou_scores[i].argmax())
        mask = masks[i][best_mask_idx].numpy().astype(bool)
        # shape: (H, W)

        # マスク領域だけ残して切り抜き（背景は白）
        # 別のnumpy配列を作成
        cropped = image_np.copy()
        # チルダ：NOT演算子
        # ~mask: maskがFalseの部分をTrueに、Trueの部分をFalseに反転
        # 該当のpixelを全て白に
        cropped[~mask] = 255

        # bbox でトリミング
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
        cropped_region = cropped[y1:y2, x1:x2]

        buf = io.BytesIO()
        Image.fromarray(cropped_region).save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        segments.append({
            "label": label,
            "score": float(score),
            "box": [float(v) for v in box.tolist()],
            "image": f"data:image/png;base64,{img_b64}",
        })

    return {"segments": segments}