import { useRef, useState } from "react";

const API_BASE = "http://localhost:8000";

type SimilarResult = {
  image: string;
  label: string;
  score: number;
};

export default function SimilarPage() {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<SimilarResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFile(f: File) {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResults([]);
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

  async function search() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults([]);
    try {
      const body = new FormData();
      body.append("file", file);
      body.append("top_k", "5");
      const res = await fetch(`${API_BASE}/similar-images`, { method: "POST", body });
      if (!res.ok) throw new Error(`サーバーエラー: ${res.status}`);
      const data = await res.json();
      setResults(data.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "不明なエラー");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <p className="subtitle">
        画像をアップロードして CIFAR-10 から類似画像 TOP 5 を検索
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
        <span className="upload-icon">🖼️</span>
        <p className="upload-hint">
          <strong>クリックして選択</strong>、またはドラッグ&ドロップ
        </p>
        <p className="upload-sub">PNG / JPG / WEBP など</p>
      </div>

      {preview && (
        <div className="preview-section">
          <img src={preview} alt="プレビュー" className="preview-img" />
          <button className="search-btn" onClick={search} disabled={loading}>
            {loading ? <><span className="spinner" />検索中...</> : "類似画像を検索"}
          </button>
          {error && <p className="error-msg">{error}</p>}
        </div>
      )}

      {results.length > 0 && (
        <div className="results-section">
          <p className="results-title">類似画像 TOP 5</p>
          <div className="results-grid">
            {results.map((item, i) => (
              <div key={i} className="result-card">
                <span className="rank-badge">#{i + 1}</span>
                <img src={item.image} alt={item.label} className="result-img" />
                <div className="result-info">
                  <p className="result-label">{item.label}</p>
                  <p className="result-score">{(item.score * 100).toFixed(1)}%</p>
                  <div className="score-bar">
                    <div
                      className="score-fill"
                      style={{ width: `${(item.score * 100).toFixed(1)}%` }}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
