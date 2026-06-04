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
      <p className="text-sm text-slate-500 mb-9">
        Grounding DINO + SAM による物体のピクセルレベル切り抜き
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
        <span className="text-[2.5rem] block mb-3">✂️</span>
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

      {segments.length > 0 && (
        <div className="flex flex-col items-center gap-4 mt-9 w-full max-w-[720px]">
          <p className="text-[0.9rem] font-semibold text-slate-400 self-start tracking-wide uppercase">
            切り抜き結果 — {segments.length} 件
          </p>
          <div className="flex flex-wrap gap-4 mt-4">
            {segments.map((seg, i) => {
              const color = COLORS[i % COLORS.length];
              return (
                <div
                  key={i}
                  className="flex flex-col items-center gap-2 border-2 rounded-[10px] p-2.5 bg-slate-800"
                  style={{ borderColor: color }}
                >
                  <img src={seg.image} alt={seg.label} className="max-w-[200px] max-h-[200px] object-contain rounded-md bg-white" />
                  <div
                    className="flex items-center gap-1.5 bg-slate-800 border border-slate-700 rounded-full px-3 py-[5px]"
                    style={{ borderColor: color }}
                  >
                    <span className="w-2 h-2 rounded-full shrink-0" style={{ background: color }} />
                    <span className="text-[0.8rem] font-semibold text-slate-200">{seg.label}</span>
                    <span className="text-[0.75rem] text-violet-600 font-bold">{(seg.score * 100).toFixed(1)}%</span>
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
