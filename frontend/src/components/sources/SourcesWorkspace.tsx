import { useEffect, useState } from "react";

import type { Source } from "../../api/types";
import { Dialog } from "../common/Dialog";
import { Button, EmptyState, ErrorState, StatusBadge } from "../common/ui";

type SourcesWorkspaceProps = {
  sources?: Source[];
  loading: boolean;
  error?: Error | null;
  uploading: boolean;
  uploadError?: Error | null;
  onUpload: (file: File) => Promise<Source>;
  onRemove: (id: string) => void;
};

export type PendingUpload = {
  id: string;
  name: string;
  createdAt: string;
  source?: Source;
};

/**
 * Merge the optimistic upload row with the server's source list.
 *
 * The upload endpoint can return before the source query has refreshed. Keep
 * the server source as the single row when it is available, and suppress a
 * newly-created duplicate while the optimistic row is still visible.
 */
export function mergePendingSource(
  sources: Source[],
  pendingUpload: PendingUpload | null,
): Source[] {
  if (!pendingUpload) {
    return sources;
  }

  const serverSource =
    pendingUpload.source ??
    sources.find(
      (source) =>
        source.document.name === pendingUpload.name &&
        source.document.created_at >= pendingUpload.createdAt,
    );
  const displayedSource: Source =
    serverSource ?? {
      document: {
        id: pendingUpload.id,
        name: pendingUpload.name,
        created_at: pendingUpload.createdAt,
      },
      ingestion: {
        id: pendingUpload.id,
        status: "uploading",
        error: null,
        created_at: pendingUpload.createdAt,
      },
    };

  return [
    displayedSource,
    ...sources.filter((source) => {
      if (source.document.id === displayedSource.document.id) {
        return false;
      }
      if (serverSource) {
        return true;
      }
      return !(
        source.document.name === pendingUpload.name &&
        source.document.created_at >= pendingUpload.createdAt
      );
    }),
  ];
}

export function SourcesWorkspace({
  sources,
  loading,
  error,
  uploading,
  uploadError,
  onUpload,
  onRemove,
}: SourcesWorkspaceProps) {
  const [file, setFile] = useState<File | null>(null);
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(null);
  const [removeId, setRemoveId] = useState<string | null>(null);

  useEffect(() => {
    if (!pendingUpload?.source) {
      return;
    }
    const sourceId = pendingUpload.source.document.id;
    if (sources?.some((source) => source.document.id === sourceId)) {
      setPendingUpload((current) =>
        current?.id === pendingUpload.id ? null : current,
      );
    }
  }, [pendingUpload, sources]);

  const selectFile = (candidate?: File): void => {
    if (!candidate || !candidate.name.toLowerCase().endsWith(".pdf")) {
      setFile(null);
      return;
    }
    setFile(candidate);
  };

  const upload = (): void => {
    if (!file) return;
    const selectedFile = file;
    const createdAt = new Date().toISOString();
    setFile(null);
    const pendingId = `uploading-${Date.now()}`;
    setPendingUpload({
      id: pendingId,
      name: selectedFile.name,
      createdAt,
    });

    // Keep the upload row visible while the server processes the document after
    // the confirmation dialog has closed.
    void onUpload(selectedFile).then(
      (source) =>
        setPendingUpload((current) =>
          current?.id === pendingId ? { ...current, source } : current,
        ),
      () =>
        setPendingUpload((current) =>
          current?.id === pendingId ? null : current,
        ),
    );
  };

  const visibleSources = mergePendingSource(sources ?? [], pendingUpload);
  const pendingSourceId = pendingUpload
    ? (pendingUpload.source?.document.id ?? pendingUpload.id)
    : null;

  return (
    <section className="sources-workspace">
      <header>
        <div>
          <h2>Sources</h2>
          <p>Manage the PDFs available to this project.</p>
        </div>
        <Button
          disabled={uploading}
          variant="primary"
          onClick={() => document.getElementById("pdf-upload")?.click()}
        >
          ＋ Add PDF
        </Button>
        <input
          id="pdf-upload"
          className="sr-only"
          type="file"
          accept="application/pdf"
          onChange={(event) => selectFile(event.target.files?.[0])}
        />
      </header>
      {uploadError && (
        <p className="form-error source-upload-error" role="alert">
          Upload failed: {uploadError.message}
        </p>
      )}
      {loading && <div className="source-skeleton">Loading sources…</div>}
      {error && <ErrorState error={error} />}
      {!loading && !error && !visibleSources.length && (
        <EmptyState title="No sources yet">
          Add PDFs to build the evidence corpus for this project.
        </EmptyState>
      )}
      {visibleSources.length ? (
        <div className="sources-table">
          <div className="sources-head">
            <span>Filename</span>
            <span>Status</span>
            <span>Added</span>
            <span />
          </div>
          {visibleSources.map((source) => (
            <div className="source-row" key={source.document.id}>
              <span className="source-name">
                {source.document.name}
              </span>
              <StatusBadge status={source.ingestion?.status ?? "pending"} />
              <time>{new Date(source.document.created_at).toLocaleDateString()}</time>
              {source.document.id === pendingSourceId ? (
                <span />
              ) : (
                <details className="action-menu source-action-menu">
                  <summary aria-label={`Manage ${source.document.name}`}>…</summary>
                  <button
                    className="danger-action"
                    onClick={() => setRemoveId(source.document.id)}
                    type="button"
                  >
                    Remove source
                  </button>
                </details>
              )}
            </div>
          ))}
        </div>
      ) : null}
      {file && (
        <Dialog title="Upload PDF" onClose={() => !uploading && setFile(null)}>
          <div className="upload-summary">
            <strong className="upload-file-name">{file.name}</strong>
            <span>{(file.size / 1_000_000).toFixed(1)} MB</span>
            <p>
              Upload begins only after confirmation. The source will then progress
              through its real ingestion stages.
            </p>
            {uploadError && (
              <p className="form-error" role="alert">
                {uploadError.message}
              </p>
            )}
            <footer>
              <Button
                variant="ghost"
                disabled={uploading}
                onClick={() => setFile(null)}
              >
                Cancel
              </Button>
              <Button
                variant="primary"
                disabled={uploading}
                onClick={upload}
              >
                {uploading ? "Uploading…" : "Upload PDF"}
              </Button>
            </footer>
          </div>
        </Dialog>
      )}
      {removeId && (
        <Dialog title="Remove source?" onClose={() => setRemoveId(null)}>
          <p>
            This removes the document from this project, but keeps reusable
            canonical content intact.
          </p>
          <footer>
            <Button variant="ghost" onClick={() => setRemoveId(null)}>
              Cancel
            </Button>
            <Button
              variant="danger"
              onClick={() => {
                onRemove(removeId);
                setRemoveId(null);
              }}
            >
              Remove source
            </Button>
          </footer>
        </Dialog>
      )}
    </section>
  );
}
