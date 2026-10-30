import { useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import type { Evidence } from "../../api/types";
import { pdfUrl } from "../../api/client";
import { Button } from "../common/ui";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();
export function PdfViewer({ evidence, onBack }: { evidence: Evidence; onBack: () => void }) { const [pages, setPages] = useState(0); const [page, setPage] = useState(evidence.page_start ?? 1); const [scale, setScale] = useState(1.15); return <section className="pdf-workspace"><header><Button variant="ghost" onClick={onBack}>← Back to evidence</Button><div><strong>{evidence.document_name}</strong><span>Page {page} / {pages || "…"}</span></div><div><Button onClick={() => setScale(Math.max(.7, scale - .15))}>−</Button><Button onClick={() => setScale(scale + .15)}>+</Button></div></header><div className="pdf-page"><Document file={pdfUrl(evidence.document_id)} loading="Loading PDF…" onLoadSuccess={({ numPages }) => setPages(numPages)} error="PDF unavailable."><Page pageNumber={page} scale={scale} /></Document></div><footer><Button disabled={page <= 1} onClick={() => setPage(page - 1)}>Previous</Button><Button disabled={pages > 0 && page >= pages} onClick={() => setPage(page + 1)}>Next</Button></footer></section>; }
