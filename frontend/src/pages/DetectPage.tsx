import { useEffect, useRef, useState } from "react";

const API_BASE = "http://localhost:8000";
const COLORS = [
  "#7c3aed", "#dc2626", "#059669", "#d97706", "#2563eb", "#db2777",
];

type Detection = {
  label: string;
  score: number;
  box: [number, number, number, number];
};

function drawResultImage(imageUrl: string, detections: Detection[]): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);

      const lineWidth = Math.max(2, img.naturalWidth / 300);
      const fontSize = Math.max(13, img.naturalWidth / 50);

      detections.forEach((det, i) => {
        const color = COLORS[i % COLORS.length];
        const [x1, y1, x2, y2] = det.box;

        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        const labelText = `${det.label}  ${(det.score * 100).toFixed(0)}%`;
        ctx.font = `bold ${fontSize}px sans-serif`;
        const textW = ctx.measureText(labelText).width;
        const padX = 6;
        const padY = 4;
        const boxH = fontSize + padY * 2;
        const labelY = y1 - boxH < 0 ? y1 + boxH : y1;

        ctx.fillStyle = color;
        ctx.fillRect(x1, labelY - boxH, textW + padX * 2, boxH);
        ctx.fillStyle = "#fff";
        ctx.fillText(labelText, x1 + padX, labelY - padY);
      });

      resolve(canvas.toDataURL());
    };
    img.src = imageUrl;
  });
}

export default function DetectPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [query, setQuery] = useState("cat .");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [resultImageUrl, setResultImageUrl] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (detections.length === 0 || !preview) {
      setResultImageUrl(null);
      return;
    }
    drawResultImage(preview, detections).then(setResultImageUrl);
  }, [detections, preview]);

  function handleFile(f: File) {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setDetections([]);
    setResultImageUrl(null);
    setError(null);
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f?.type.startsWith("image/")) handleFile(f);
  }

  async function detect() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setDetections([]);
    try {
      const normalizedQuery = query.trimEnd().endsWith(".")
        ? query
        : query.trimEnd() + " .";
      const body = new FormData();
      body.append("file", file);
      body.append("labels", normalizedQuery);
      const res = await fetch(`${API_BASE}/detect`, { method: "POST", body });
      if (!res.ok) throw new Error(`サーバーエラー: ${res.status}`);
      const data = await res.json();
      setDetections(data.detections);
    } catch (err) {
      setError(err instanceof Error ? err.message : "不明なエラー");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <p className="subtitle">
        画像とテキストクエリを入力して物体を検出（Grounding DINO）
      </p>

      <div
        className={`drop-zone ${dragging ? "drag-over" : ""}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="file-input"
          onChange={onInputChange}
        />
        <span className="upload-icon">🔍</span>
        <p className="upload-hint">
          <strong>クリックして選択</strong>、またはドラッグ&ドロップ
        </p>
        <p className="upload-sub">PNG / JPG / WEBP など</p>
      </div>

      {preview && (
        <div className="preview-section">
          <img src={preview} alt="プレビュー" className="preview-img" />
          <div className="query-row">
            <input
              type="text"
              className="query-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="例: cat . dog . person ."
              onKeyDown={(e) => e.key === "Enter" && detect()}
            />
            <button className="search-btn" onClick={detect} disabled={loading}>
              {loading ? <><span className="spinner" />検出中...</> : "検出"}
            </button>
          </div>
          <p className="query-hint">
            ラベルは <code> . </code> で区切り、末尾に <code>.</code> を付けてください
          </p>
          {error && <p className="error-msg">{error}</p>}
        </div>
      )}

      {resultImageUrl && detections.length > 0 && (
        <div className="detect-result">
          <p className="results-title">
            検出結果 — {detections.length} 件
          </p>
          <img src={resultImageUrl} className="detect-canvas" alt="detection result" />
          <div className="detect-labels">
            {detections.map((det, i) => {
              const color = COLORS[i % COLORS.length];
              return (
                <div
                  key={i}
                  className="detect-tag"
                  style={{ borderColor: color }}
                >
                  <span
                    className="detect-tag-dot"
                    style={{ background: color }}
                  />
                  <span className="detect-tag-label">{det.label}</span>
                  <span className="detect-tag-score">
                    {(det.score * 100).toFixed(1)}%
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );
}
