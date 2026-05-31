import { NavLink, Route, Routes } from "react-router-dom";
import "./App.css";
import SimilarPage from "./pages/SimilarPage";
import DetectPage from "./pages/DetectPage";

export default function App() {
  return (
    <div className="page">
      <h1 className="title">ViT Visualizer</h1>

      <nav className="nav">
        <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
          類似画像検索
        </NavLink>
        <NavLink to="/detect" className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
          物体検出(rectangle)
        </NavLink>
      </nav>

      <Routes>
        <Route path="/" element={<SimilarPage />} />
        <Route path="/detect" element={<DetectPage />} />
      </Routes>
    </div>
  );
}
