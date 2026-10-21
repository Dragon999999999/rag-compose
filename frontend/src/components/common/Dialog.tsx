import { useEffect, useState, type FormEvent, type PropsWithChildren } from "react";
import type { Pipeline, PipelineCompatibility } from "../../api/types";
import { Button } from "./ui";

export function Dialog({ title, onClose, children }: PropsWithChildren<{ title: string; onClose: () => void }>) { useEffect(() => { const close = (event: KeyboardEvent) => event.key === "Escape" && onClose(); window.addEventListener("keydown", close); return () => window.removeEventListener("keydown", close); }, [onClose]); return <div className="dialog-backdrop" role="presentation" onMouseDown={onClose}><section className="dialog" role="dialog" aria-modal="true" aria-label={title} onMouseDown={event => event.stopPropagation()}><header><h2>{title}</h2><Button variant="ghost" aria-label="Close dialog" onClick={onClose}>×</Button></header>{children}</section></div>; }
export function NameDialog({ title, initialName = "", submitLabel, pending, error, onClose, onSubmit }: { title: string; initialName?: string; submitLabel: string; pending?: boolean; error?: Error | null; onClose: () => void; onSubmit: (name: string) => void }) { const [name, setName] = useState(initialName); const submit = (event: FormEvent) => { event.preventDefault(); if (name.trim()) onSubmit(name.trim()); }; return <Dialog title={title} onClose={onClose}><form className="dialog-form" onSubmit={submit}><label>Name<input autoFocus value={name} maxLength={255} onChange={event => setName(event.target.value)} /></label>{error && <p className="form-error" role="alert">{error.message}</p>}<footer><Button type="button" variant="ghost" onClick={onClose}>Cancel</Button><Button type="submit" variant="primary" disabled={pending || !name.trim()}>{pending ? "Saving…" : submitLabel}</Button></footer></form></Dialog>; }

export function ProjectCreateDialog({ pipelines, pending, error, onClose, onSubmit }: { pipelines: Pipeline[]; pending?: boolean; error?: Error | null; onClose: () => void; onSubmit: (name: string, ingestionPipelineId?: string) => void }) {
  const [name, setName] = useState("");
  const [pipelineId, setPipelineId] = useState("");
  const submit = (event: FormEvent): void => {
    event.preventDefault();
    if (name.trim()) onSubmit(name.trim(), pipelineId || undefined);
  };
  return <Dialog title="Create project" onClose={onClose}><form className="dialog-form" onSubmit={submit}>
    <label>Name<input autoFocus value={name} maxLength={255} onChange={event => setName(event.target.value)} /></label>
    <label>Ingestion pipeline<select value={pipelineId} onChange={event => setPipelineId(event.target.value)}><option value="">Application default</option>{pipelines.map(pipeline => <option key={pipeline.id} value={pipeline.id}>{pipeline.name} (v{pipeline.version})</option>)}</select></label>
    {error && <p className="form-error" role="alert">{error.message}</p>}
    <footer><Button type="button" variant="ghost" onClick={onClose}>Cancel</Button><Button type="submit" variant="primary" disabled={pending || !name.trim()}>{pending ? "Creating…" : "Create project"}</Button></footer>
  </form></Dialog>;
}

export function ChatCreateDialog({ compatibility, compatibilityError, pending, error, onClose, onSubmit }: { compatibility: PipelineCompatibility[]; compatibilityError?: Error | null; pending?: boolean; error?: Error | null; onClose: () => void; onSubmit: (name: string, pipelineId?: string) => void }) {
  const [name, setName] = useState("New chat");
  const [pipelineId, setPipelineId] = useState("");
  const submit = (event: FormEvent): void => {
    event.preventDefault();
    if (name.trim()) onSubmit(name.trim(), pipelineId || undefined);
  };
  const compatible = compatibility.filter(item => item.compatible);
  return <Dialog title="New chat" onClose={onClose}><form className="dialog-form" onSubmit={submit}>
    <label>Name<input autoFocus value={name} maxLength={255} onChange={event => setName(event.target.value)} /></label>
    <label>Query pipeline<select value={pipelineId} onChange={event => setPipelineId(event.target.value)}><option value="">Application default</option>{compatible.map(item => <option key={item.pipeline.id} value={item.pipeline.id}>{item.pipeline.name} (v{item.pipeline.version})</option>)}{compatibility.filter(item => !item.compatible).map(item => <option disabled key={item.pipeline.id} value={item.pipeline.id}>{item.pipeline.name} — unavailable</option>)}</select></label>
    {compatibilityError && <p className="form-error" role="alert">Unable to load project compatibility: {compatibilityError.message}</p>}
    {compatibility.filter(item => !item.compatible).map(item => <p className="compatibility-option-warning" key={`reason-${item.pipeline.id}`}>Unavailable: <strong>{item.pipeline.name}</strong> — {item.reason ?? `Requires ${item.missing_capabilities.join(", ")}`}</p>)}
    {error && <p className="form-error" role="alert">{error.message}</p>}
    <footer><Button type="button" variant="ghost" onClick={onClose}>Cancel</Button><Button type="submit" variant="primary" disabled={pending || !name.trim()}>{pending ? "Creating…" : "Create chat"}</Button></footer>
  </form></Dialog>;
}
