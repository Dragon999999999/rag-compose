import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./app/App";
import { Providers } from "./app/providers";
import "./styles/index.css";
createRoot(document.getElementById("root")!).render(<Providers><BrowserRouter><App /></BrowserRouter></Providers>);
