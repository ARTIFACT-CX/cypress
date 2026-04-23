import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import { ACTIVE_THEME, applyTheme, themes } from "./themes";

applyTheme(themes[ACTIVE_THEME]);

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
