import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import { ToastProvider } from "./components/Toast";
import { ACTIVE_THEME, applyTheme, themes } from "./themes";
import { initServerStore } from "./store/bootstrap";

applyTheme(themes[ACTIVE_THEME]);
// Boot the server-store listeners + poller exactly once before the
// first render. Idempotent against StrictMode's double-mount.
initServerStore();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {/* ToastProvider wraps the whole app so any component can call useToast(). */}
    <ToastProvider>
      <App />
    </ToastProvider>
  </React.StrictMode>,
);
