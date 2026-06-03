import { useRef, useState } from "react";

const API_BASE = "http://localhost:8000";
const COLORS = [
  "#7c3aed", "#dc2626", "#059669", "#d97706", "#2563eb", "#db2777",
];

type Segment = {
  label: string;
  score: number;
  box: [number, number, number, number];
  image: string;
};

export default function DetectPixelPage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [query, setQuery] = useState("cat .");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFile(f: File) {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setSegments([]);
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
    setSegments([]);
    try {
      const normalizedQuery = query.trimEnd().endsWith(".")
        ? query
        : query.trimEnd() + " .";
      const body = new FormData();
      body.append("file", file);
      body.append("labels", normalizedQuery);
      const res = await fetch(`${API_BASE}/detect_with_pixel`, { method: "POST", body });
      if (!res.ok) throw new Error(`サーバーエラー: ${res.status}`);
      const data = await res.json();
      setSegments(data.segments);
    } catch (err) {
      setError(err instanceof Error ? err.message : "不明なエラー");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <p className="subtitle">
        Grounding DINO + SAM による物体のピクセルレベル切り抜き
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
        <span className="upload-icon">✂️</span>
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

      {segments.length > 0 && (
        <div className="detect-result">
          <p className="results-title">
            切り抜き結果 — {segments.length} 件
          </p>
          <div className="pixel-segments">
            {segments.map((seg, i) => {
              const color = COLORS[i % COLORS.length];
              return (
                <div key={i} className="pixel-segment-card" style={{ borderColor: color }}>
                  <img src={seg.image} alt={seg.label} className="pixel-segment-img" />
                  <div className="detect-tag" style={{ borderColor: color }}>
                    <span className="detect-tag-dot" style={{ background: color }} />
                    <span className="detect-tag-label">{seg.label}</span>
                    <span className="detect-tag-score">{(seg.score * 100).toFixed(1)}%</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </>
  );
}
