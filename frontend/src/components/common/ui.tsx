import type { ButtonHTMLAttributes, PropsWithChildren } from "react";

export function Button({ className = "", variant = "secondary", ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger" }) { return <button className={`button button-${variant} ${className}`} {...props} />; }
export function IconButton(props: ButtonHTMLAttributes<HTMLButtonElement>) { return <Button variant="ghost" className="icon-button" {...props} />; }
export function EmptyState({ title, children }: PropsWithChildren<{ title: string }>) { return <div className="empty-state"><h2>{title}</h2><p>{children}</p></div>; }
export function ErrorState({ error, retry }: { error: Error; retry?: () => void }) { return <div className="error-state" role="alert"><strong>Something went wrong</strong><p>{error.message}</p>{retry && <Button onClick={retry}>Try again</Button>}</div>; }
export function StatusBadge({ status }: { status: string }) { return <span className={`status status-${status.toLowerCase()}`}>{status}</span>; }
