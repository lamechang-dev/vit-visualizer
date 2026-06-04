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
      <p className="text-sm text-slate-500 mb-9">
        画像をアップロードして CIFAR-10 から類似画像 TOP 5 を検索
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
        <span className="text-[2.5rem] block mb-3">🖼️</span>
        <p className="text-[0.9rem] text-slate-400">
          <strong className="text-indigo-400">クリックして選択</strong>、またはドラッグ&ドロップ
        </p>
        <p className="text-[0.78rem] text-slate-600 mt-1.5">PNG / JPG / WEBP など</p>
      </div>

      {preview && (
        <div className="flex flex-col items-center gap-3.5 mt-7">
          <img src={preview} alt="プレビュー" className="w-40 h-40 object-cover rounded-xl border-2 border-slate-700" />
          <button
            className="px-7 py-2.5 bg-linear-to-br from-violet-600 to-blue-600 text-white border-none rounded-lg text-[0.9rem] font-semibold cursor-pointer transition-[opacity,transform] flex items-center gap-1.5 hover:opacity-90 active:scale-[0.97] disabled:opacity-45 disabled:cursor-not-allowed"
            onClick={search}
            disabled={loading}
          >
            {loading ? (
              <><span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />検索中...</>
            ) : "類似画像を検索"}
          </button>
          {error && <p className="text-[0.85rem] text-red-400">{error}</p>}
        </div>
      )}

      {results.length > 0 && (
        <div className="flex flex-col items-center gap-5 mt-11 w-full max-w-[600px]">
          <p className="text-[0.9rem] font-semibold text-slate-400 self-start tracking-wide uppercase">
            類似画像 TOP 5
          </p>
          <div className="grid grid-cols-5 gap-3 w-full">
            {results.map((item, i) => (
              <div
                key={i}
                className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden flex flex-col items-center transition-[transform,border-color] hover:-translate-y-[3px] hover:border-violet-600"
              >
                <span className="self-start bg-violet-600/20 text-violet-400 text-[0.68rem] font-bold py-0.5 px-[7px] mt-[7px] ml-[7px] mb-[3px] rounded">
                  #{i + 1}
                </span>
                <img src={item.image} alt={item.label} className="w-full aspect-square object-cover [image-rendering:pixelated]" />
                <div className="p-2 w-full text-center">
                  <p className="text-[0.72rem] font-semibold text-slate-200 capitalize whitespace-nowrap overflow-hidden text-ellipsis">
                    {item.label}
                  </p>
                  <p className="text-[0.68rem] text-slate-500 mt-0.5">{(item.score * 100).toFixed(1)}%</p>
                  <div className="w-full h-[3px] bg-[#0f1117] rounded-sm mt-[5px] overflow-hidden">
                    <div
                      className="h-full rounded-sm bg-linear-to-r from-violet-600 to-blue-600"
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
