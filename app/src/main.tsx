import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import { ToastProvider } from "./components/Toast";
import { ACTIVE_THEME, applyTheme, themes } from "./themes";

applyTheme(themes[ACTIVE_THEME]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {/* ToastProvider wraps the whole app so any component can call useToast(). */}
    <ToastProvider>
      <App />
    </ToastProvider>
  </React.StrictMode>,
);
