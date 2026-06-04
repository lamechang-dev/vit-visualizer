import { NavLink, Route, Routes } from "react-router-dom";
import SimilarPage from "./pages/SimilarPage";
import DetectPage from "./pages/DetectPage";
import DetectPixelPage from "./pages/DetectPixelPage";

export default function App() {
  return (
    <div className="flex flex-col items-center px-6 pb-20 pt-12 min-h-screen bg-[#0f1117] text-slate-200">
      <h1 className="text-[1.75rem] font-bold tracking-[-0.5px] bg-linear-to-br from-violet-600 to-blue-600 bg-clip-text text-transparent mb-2">
        ViT Visualizer
      </h1>

      <nav className="flex gap-1 bg-slate-800 rounded-xl p-1 mb-7">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `px-6 py-2 rounded-[7px] text-sm font-semibold no-underline transition-[background,color] ${
              isActive
                ? "bg-linear-to-br from-violet-600 to-blue-600 text-white"
                : "text-slate-500 hover:text-slate-400"
            }`
          }
        >
          類似画像検索
        </NavLink>
        <NavLink
          to="/detect"
          className={({ isActive }) =>
            `px-6 py-2 rounded-[7px] text-sm font-semibold no-underline transition-[background,color] ${
              isActive
                ? "bg-linear-to-br from-violet-600 to-blue-600 text-white"
                : "text-slate-500 hover:text-slate-400"
            }`
          }
        >
          物体検出(rectangle)
        </NavLink>
        <NavLink
          to="/detect_with_pixel"
          className={({ isActive }) =>
            `px-6 py-2 rounded-[7px] text-sm font-semibold no-underline transition-[background,color] ${
              isActive
                ? "bg-linear-to-br from-violet-600 to-blue-600 text-white"
                : "text-slate-500 hover:text-slate-400"
            }`
          }
        >
          物体検出(pixel)
        </NavLink>
      </nav>

      <Routes>
        <Route path="/" element={<SimilarPage />} />
        <Route path="/detect" element={<DetectPage />} />
        <Route path="/detect_with_pixel" element={<DetectPixelPage />} />
      </Routes>
    </div>
  );
}
