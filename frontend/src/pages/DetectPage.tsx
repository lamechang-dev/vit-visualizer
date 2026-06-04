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
      <p className="text-sm text-slate-500 mb-9">
        画像とテキストクエリを入力して物体を検出（Grounding DINO）
      </p>

      <div
        className={`relative w-full max-w-[480px] border-2 border-dashed rounded-2xl py-10 px-6 text-center cursor-pointer transition-[border-color,background] ${
          dragging
            ? "border-violet-600 bg-violet-600/[0.06]"
            : "border-slate-700 hover:border-slate-600"
        }`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
          onChange={onInputChange}
        />
        <span className="text-[2.5rem] block mb-3">🔍</span>
        <p className="text-[0.9rem] text-slate-400">
          <strong className="text-indigo-400">クリックして選択</strong>、またはドラッグ&ドロップ
        </p>
        <p className="text-[0.78rem] text-slate-600 mt-1.5">PNG / JPG / WEBP など</p>
      </div>

      {preview && (
        <div className="flex flex-col items-center gap-3.5 mt-7">
          <img src={preview} alt="プレビュー" className="w-40 h-40 object-cover rounded-xl border-2 border-slate-700" />
          <div className="flex gap-2 w-full max-w-[480px]">
            <input
              type="text"
              className="flex-1 px-3.5 py-2.5 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 text-[0.9rem] outline-none transition-[border-color] focus:border-violet-600 placeholder:text-slate-600"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="例: cat . dog . person ."
              onKeyDown={(e) => e.key === "Enter" && detect()}
            />
            <button
              className="px-7 py-2.5 bg-linear-to-br from-violet-600 to-blue-600 text-white border-none rounded-lg text-[0.9rem] font-semibold cursor-pointer transition-[opacity,transform] flex items-center gap-1.5 hover:opacity-90 active:scale-[0.97] disabled:opacity-45 disabled:cursor-not-allowed"
              onClick={detect}
              disabled={loading}
            >
              {loading ? (
                <><span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />検出中...</>
              ) : "検出"}
            </button>
          </div>
          <p className="text-[0.75rem] text-slate-600">
            ラベルは <code className="bg-slate-800 px-[5px] py-px rounded text-indigo-400 font-mono"> . </code> で区切り、末尾に <code className="bg-slate-800 px-[5px] py-px rounded text-indigo-400 font-mono">.</code> を付けてください
          </p>
          {error && <p className="text-[0.85rem] text-red-400">{error}</p>}
        </div>
      )}

      {resultImageUrl && detections.length > 0 && (
        <div className="flex flex-col items-center gap-4 mt-9 w-full max-w-[720px]">
          <p className="text-[0.9rem] font-semibold text-slate-400 self-start tracking-wide uppercase">
            検出結果 — {detections.length} 件
          </p>
          <img src={resultImageUrl} className="w-full h-auto rounded-xl border-2 border-slate-700" alt="detection result" />
          <div className="flex flex-wrap gap-2 justify-center">
            {detections.map((det, i) => {
              const color = COLORS[i % COLORS.length];
              return (
                <div
                  key={i}
                  className="flex items-center gap-1.5 bg-slate-800 border border-slate-700 rounded-full px-3 py-[5px]"
                  style={{ borderColor: color }}
                >
                  <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
                  <span className="text-[0.8rem] font-semibold text-slate-200">{det.label}</span>
                  <span className="text-[0.75rem] text-violet-600 font-bold">
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
