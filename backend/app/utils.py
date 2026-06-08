import tempfile
import cv2
import torch
from typing import TypedDict

# CUDA: Compute Unified Device Architecture
# mps: Metal Performance Shaders(Apple製 GPU)
# cpu: 通常のCPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Frame(TypedDict):
    index: int
    timestamp_sec: float
    image: str


# target_fps: 5なら、1秒に5フレーム(画像)を抜き出し
def extract_frames(video_bytes: bytes, target_fps: int = 5) -> list[Frame]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    # VideoCaptureは動画を「フレームの列」として読み込む
    cap = cv2.VideoCapture(tmp_path)
    # 元のビデオのFPSを取得
    # 30なら、1秒に30フレーム
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    # ex: 30.0
    print("source_fps:", source_fps)
    # 元のFPSをtarget_fpsに合わせるための間隔を計算
    interval = max(1, round(source_fps / target_fps))

    frames: list[Frame] = []
    frame_idx = 0

    while cap.isOpened():
        # 1フレーム読み込む
        # ret: 読み込み成功かどうか。動画の最後まで読み込んだらFalseになる
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            # OpenCVはBGRなのでRGBに変換してからエンコード
            # (高さ、幅、チャンネル)
            # (240, 320, 3)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            success, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if success:
                import base64
                frames.append({
                    "index": len(frames),
                    "timestamp_sec": round(frame_idx / source_fps, 3),
                    "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
                })
        frame_idx += 1

    cap.release()
    return frames
