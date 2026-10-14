import { useState } from "react";
import type { AnalysisResult } from "./api/types";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";

type View = { name: "upload" } | { name: "result"; result: AnalysisResult };

export default function App() {
  const [view, setView] = useState<View>({ name: "upload" });

  return (
    <div className="app">
      <header className="app-header">
        <h1>Facial Index Classifier</h1>
        <p className="subtitle">
          Video-based Facial Index measurement prototype — not a medical device.
        </p>
      </header>
      {view.name === "upload" ? (
        <UploadPage onCompleted={(r) => setView({ name: "result", result: r })} />
      ) : (
        <ResultPage
          initial={view.result}
          onBack={() => setView({ name: "upload" })}
        />
      )}
      <footer className="app-footer">
        <p>
          Landmarks are MediaPipe baseline candidates only and require expert
          confirmation. See ARCHITECTURE.md §4.
        </p>
      </footer>
    </div>
  );
}
