import { useState } from "react";

import type { Source } from "../../api/types";
import { Dialog } from "../common/Dialog";
import { Button, EmptyState, ErrorState, StatusBadge } from "../common/ui";

type SourcesWorkspaceProps = {
  sources?: Source[];
  loading: boolean;
  error?: Error | null;
  uploading: boolean;
  uploadError?: Error | null;
  onUpload: (file: File) => Promise<unknown>;
  onRemove: (id: string) => void;
  onOpen: (source: Source) => void;
  onBack: () => void;
};

export function SourcesWorkspace({
  sources,
  loading,
  error,
  uploading,
  uploadError,
  onUpload,
  onRemove,
  onOpen,
  onBack,
}: SourcesWorkspaceProps) {
  const [file, setFile] = useState<File | null>(null);
  const [removeId, setRemoveId] = useState<string | null>(null);

  const selectFile = (candidate?: File): void => {
    if (!candidate || !candidate.name.toLowerCase().endsWith(".pdf")) {
      setFile(null);
      return;
    }
    setFile(candidate);
  };

  const upload = async (): Promise<void> => {
    if (!file) return;
    try {
      await onUpload(file);
      setFile(null);
    } catch {
      // The mutation error is rendered in the dialog without exposing a stack trace.
    }
  };

  return <section className="sources-workspace"><header><div><Button variant="ghost" onClick={onBack}>← Back to chat</Button><h2>Sources</h2><p>Manage the PDFs available to this project.</p></div><Button variant="primary" onClick={() => document.getElementById("pdf-upload")?.click()}>＋ Add PDF</Button><input id="pdf-upload" className="sr-only" type="file" accept="application/pdf" onChange={event => selectFile(event.target.files?.[0])} /></header>{loading && <div className="source-skeleton">Loading sources…</div>}{error && <ErrorState error={error} />}{!loading && !error && !sources?.length && <EmptyState title="No sources yet">Add PDFs to build the evidence corpus for this project.</EmptyState>}{sources?.length ? <div className="sources-table"><div className="sources-head"><span>Filename</span><span>Status</span><span>Added</span><span /></div>{sources.map(source => <div className="source-row source-row-openable" key={source.document.id}><button className="source-name" onClick={() => onOpen(source)}>{source.document.name}</button><StatusBadge status={source.ingestion?.status ?? "pending"} /><time>{new Date(source.document.created_at).toLocaleDateString()}</time><details className="action-menu"><summary aria-label={`Manage ${source.document.name}`}>…</summary><button className="danger-action" onClick={() => setRemoveId(source.document.id)} type="button">Remove source</button></details></div>)}</div> : null}{file && <Dialog title="Upload PDF" onClose={() => !uploading && setFile(null)}><div className="upload-summary"><strong>{file.name}</strong><span>{(file.size / 1_000_000).toFixed(1)} MB</span><p>Upload begins only after confirmation. The source will then progress through its real ingestion stages.</p>{uploadError && <p className="form-error" role="alert">{uploadError.message}</p>}<footer><Button variant="ghost" disabled={uploading} onClick={() => setFile(null)}>Cancel</Button><Button variant="primary" disabled={uploading} onClick={() => void upload()}>{uploading ? "Uploading…" : "Upload PDF"}</Button></footer></div></Dialog>}{removeId && <Dialog title="Remove source?" onClose={() => setRemoveId(null)}><p>This removes the document from this project, but keeps reusable canonical content intact.</p><footer><Button variant="ghost" onClick={() => setRemoveId(null)}>Cancel</Button><Button variant="danger" onClick={() => { onRemove(removeId); setRemoveId(null); }}>Remove source</Button></footer></Dialog>}</section>;
}
