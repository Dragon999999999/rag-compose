import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";

import { isNotFoundError } from "../../api/client";
import type { GroundedAnswer, Message } from "../../api/types";
import { useEvidence } from "../../features/hooks";
import { Button, EmptyState, ErrorState } from "../common/ui";

type ConversationTurn = {
  id: string;
  prompt: Message;
  response?: Message;
  liveAnswer?: GroundedAnswer;
  runId?: string;
  title: string;
};

type ChatWorkspaceProps = {
  chatId: string;
  messages?: Message[];
  answer: GroundedAnswer | null;
  loading: boolean;
  error?: Error | null;
  generationError?: Error | null;
  failedPrompt?: string | null;
  activeChunk: string | null;
  onSelectAnswer: (runId: string, turnTitle?: string) => void;
  onSelectCitation: (chunkId: string) => void;
  onAsk: (message: string) => void;
};

function loadDraft(chatId: string): string {
  return window.sessionStorage.getItem(`chat-draft:${chatId}`) ?? "";
}

function saveDraft(chatId: string, value: string): void {
  window.sessionStorage.setItem(`chat-draft:${chatId}`, value);
}

/** Render a chat as navigable prompt-and-answer turns. */
export function ChatWorkspace({
  chatId,
  messages,
  answer,
  loading,
  error,
  generationError,
  failedPrompt,
  activeChunk,
  onSelectAnswer,
  onSelectCitation,
  onAsk,
}: ChatWorkspaceProps) {
  const [value, setValue] = useState(() => loadDraft(chatId));
  const [hoveredTurnTitle, setHoveredTurnTitle] = useState<string | null>(null);
  const thread = useRef<HTMLDivElement>(null);
  const messagesHydrated = useRef(false);
  const turnElements = useRef(new Map<string, HTMLElement>());
  const turns = useMemo(
    () => buildConversationTurns(messages ?? [], answer),
    [messages, answer],
  );
  const completedTurns = turns.filter(
    (turn) => turn.response !== undefined || turn.liveAnswer !== undefined,
  );
  const failedTurn = failedPrompt
    ? [...turns]
        .reverse()
        .find((turn) => turn.prompt.content === failedPrompt && !turn.response)
    : undefined;
  const hasInlineGenerationFailure = Boolean(generationError && failedPrompt);
  const showTurnNavigation = completedTurns.length > 1;

  useEffect(() => {
    setValue(loadDraft(chatId));
  }, [chatId]);

  useEffect(() => {
    saveDraft(chatId, value);
  }, [chatId, value]);

  const scrollToLatest = (): void =>
    thread.current?.scrollTo({
      top: thread.current.scrollHeight,
      behavior: "smooth",
    });

  useEffect(() => {
    if (messages === undefined) {
      return;
    }
    if (!messagesHydrated.current) {
      messagesHydrated.current = true;
      const savedPosition = Number(
        window.sessionStorage.getItem(`chat-scroll:${chatId}`) ?? "0",
      );
      requestAnimationFrame(() => {
        thread.current?.scrollTo({
          top: Number.isFinite(savedPosition) ? savedPosition : 0,
        });
      });
      return;
    }
    scrollToLatest();
  }, [chatId, loading, messages, messages?.length]);

  useEffect(
    () => () => {
      window.sessionStorage.setItem(
        `chat-scroll:${chatId}`,
        String(thread.current?.scrollTop ?? 0),
      );
    },
    [chatId],
  );

  const send = (): void => {
    if (value.trim()) {
      onAsk(value.trim());
      saveDraft(chatId, "");
      setValue("");
      requestAnimationFrame(scrollToLatest);
    }
  };

  const submit = (event: FormEvent): void => {
    event.preventDefault();
    send();
  };

  const scrollToTurn = (turn: ConversationTurn): void => {
    if (turn.runId) {
      onSelectAnswer(turn.runId, turn.title);
    }
    turnElements.current
      .get(turn.id)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  useEffect(() => {
    const root = thread.current;
    if (!root || completedTurns.length < 2) {
      return;
    }
    const turnsById = new Map(
      turns.filter((turn) => turn.runId).map((turn) => [turn.id, turn]),
    );
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
        const turnId = visible?.target.getAttribute("data-turn-id");
        const turn = turnId ? turnsById.get(turnId) : undefined;
        if (turn?.runId) {
          onSelectAnswer(turn.runId, turn.title);
        }
      },
      { root, threshold: [0.35, 0.65] },
    );
    for (const turn of turnsById.values()) {
      const element = turnElements.current.get(turn.id);
      if (element) {
        observer.observe(element);
      }
    }
    return () => observer.disconnect();
  }, [onSelectAnswer, turns]);

  return (
    <section className="chat-workspace">
      <div
        className={`chat-thread ${showTurnNavigation ? "with-turn-navigation" : ""}`}
        ref={thread}
      >
        {showTurnNavigation && (
          <nav aria-label="Conversation turns" className="turn-navigation">
            {completedTurns.map((turn) => (
              <button
                aria-label={`Jump to ${turn.title}`}
                className="turn-marker"
                key={turn.id}
                onBlur={() => setHoveredTurnTitle(null)}
                onClick={() => scrollToTurn(turn)}
                onFocus={() => setHoveredTurnTitle(turn.title)}
                onMouseEnter={() => setHoveredTurnTitle(turn.title)}
                onMouseLeave={() => setHoveredTurnTitle(null)}
                type="button"
                title={turn.title}
              >
                <span aria-hidden="true" className="turn-marker-line" />
              </button>
            ))}
          </nav>
        )}
        <div className="conversation-content">
          {error && !hasInlineGenerationFailure && <ErrorState error={error} />}
          {!loading && !turns.length && (
            <EmptyState title="No chats yet">
              Start a new conversation to ask grounded questions about this project.
            </EmptyState>
          )}
          {turns.map((turn) => (
            <section
              className="conversation-turn"
              data-turn-id={turn.id}
              key={turn.id}
              ref={(element) => {
                if (element) {
                  turnElements.current.set(turn.id, element);
                } else {
                  turnElements.current.delete(turn.id);
                }
              }}
            >
              <div className="conversation-turn-content">
                <MessageArticle
                  message={turn.prompt}
                  onRetry={() => onAsk(turn.prompt.content)}
                  retryDisabled={loading}
                />
                {failedTurn?.id === turn.id && generationError && (
                  <GenerationFailure
                    error={generationError}
                    onRetry={() => onAsk(turn.prompt.content)}
                  />
                )}
                {turn.liveAnswer ? (
                  <AssistantAnswer
                    activeChunk={activeChunk}
                    answer={turn.liveAnswer}
                    onSelectAnswer={onSelectAnswer}
                    onSelectCitation={onSelectCitation}
                  />
                ) : turn.response ? (
                  <PersistedAssistantAnswer
                    activeChunk={activeChunk}
                    message={turn.response}
                    onSelectAnswer={onSelectAnswer}
                    onSelectCitation={onSelectCitation}
                  />
                ) : null}
              </div>
            </section>
          ))}
          {failedPrompt && !failedTurn && generationError && (
            <section className="conversation-turn failed-conversation-turn">
              <div className="conversation-turn-content">
                <MessageArticle
                  message={createPendingMessage(chatId, failedPrompt)}
                  onRetry={() => onAsk(failedPrompt)}
                  retryDisabled={loading}
                />
                <GenerationFailure
                  error={generationError}
                  onRetry={() => onAsk(failedPrompt)}
                />
              </div>
            </section>
          )}
          {loading && (
            <article className="message message-assistant generation">
              <span className="message-role">Assistant</span>
              <span>Generating grounded answer…</span>
            </article>
          )}
        </div>
      </div>
      {hoveredTurnTitle && (
        <span className="turn-navigation-tooltip" role="tooltip">
          {hoveredTurnTitle}
        </span>
      )}
      <form className="composer" onSubmit={submit}>
        <label className="sr-only" htmlFor="question">
          Ask about this project
        </label>
        <textarea
          autoFocus
          disabled={loading}
          id="question"
          onChange={(event) => {
            const nextValue = event.target.value;
            saveDraft(chatId, nextValue);
            setValue(nextValue);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send();
            }
          }}
          placeholder="Ask about this project…"
          value={value}
        />
        <Button variant="primary" disabled={loading || !value.trim()}>
          {loading ? "Sending…" : "Send"}
        </Button>
      </form>
    </section>
  );
}

function buildConversationTurns(
  messages: Message[],
  answer: GroundedAnswer | null,
): ConversationTurn[] {
  // Pair each user prompt with its following assistant answer.
  const turns: ConversationTurn[] = [];
  let currentTurn: ConversationTurn | undefined;

  for (const message of messages) {
    if (message.role === "user") {
      currentTurn = {
        id: `turn-${message.id}`,
        prompt: message,
        title: fallbackTurnTitle(message.content),
      };
      turns.push(currentTurn);
      continue;
    }

    if (message.role === "assistant" && currentTurn && !currentTurn.response) {
      currentTurn.response = message;
      currentTurn.runId = message.rag_run_id ?? undefined;
      currentTurn.title = message.turn_title ?? currentTurn.title;
      if (isCurrentAnswer(message, answer)) {
        currentTurn.liveAnswer = answer;
        currentTurn.runId = answer.rag_run_id;
        currentTurn.title = answer.turn_title ?? currentTurn.title;
      }
    }
  }

  if (answer && !turns.some((turn) => turn.liveAnswer)) {
    const latestTurn = turns.at(-1);
    if (latestTurn && !latestTurn.response) {
      latestTurn.liveAnswer = answer;
      latestTurn.runId = answer.rag_run_id;
      latestTurn.title = answer.turn_title ?? latestTurn.title;
    }
  }

  return turns;
}

function fallbackTurnTitle(prompt: string): string {
  // Label older messages that predate model-generated navigation titles.
  const normalized = prompt.replace(/\s+/g, " ").trim();
  return normalized.length <= 80
    ? normalized
    : `${normalized.slice(0, 77).trimEnd()}…`;
}

function createPendingMessage(chatId: string, content: string): Message {
  return {
    id: `failed-${chatId}`,
    chat_id: chatId,
    role: "user",
    content,
    created_at: new Date().toISOString(),
    rag_run_id: null,
    turn_title: null,
  };
}

function isCurrentAnswer(
  message: Message,
  answer: GroundedAnswer | null,
): answer is GroundedAnswer {
  // Identify the persisted assistant message represented by a live answer.
  return Boolean(
    answer &&
      (answer.message_id === message.id || answer.rag_run_id === message.rag_run_id),
  );
}

function MessageArticle({
  message,
  onRetry,
  retryDisabled = false,
}: {
  message: Message;
  onRetry?: () => void;
  retryDisabled?: boolean;
}) {
  // Render a user prompt without treating conversation history as evidence.
  return (
    <article className={`message message-${message.role}`}>
      <div className="message-heading">
        <span className="message-role">You</span>
        <div className="message-actions">
          {onRetry && (
            <RetryButton
              disabled={retryDisabled}
              onRetry={onRetry}
              prompt={message.content}
            />
          )}
          <CopyButton label="Copy prompt" text={message.content} />
        </div>
      </div>
      <div className="message-body">{message.content}</div>
    </article>
  );
}

function GenerationFailure({
  error,
  onRetry,
}: {
  error: Error;
  onRetry: () => void;
}) {
  return (
    <aside className="generation-failure" role="alert">
      <span className="generation-failure-flag" aria-hidden="true">
        !
      </span>
      <div>
        <strong>Response generation failed</strong>
        <p>There was an issue generating a response for this prompt.</p>
      </div>
      <Button
        className="generation-failure-retry"
        onClick={(event) => {
          event.stopPropagation();
          onRetry();
        }}
      >
        Retry
      </Button>
      <span className="sr-only">{error.message}</span>
    </aside>
  );
}

function PersistedAssistantAnswer({
  message,
  activeChunk,
  onSelectAnswer,
  onSelectCitation,
}: {
  message: Message;
  activeChunk: string | null;
  onSelectAnswer: (runId: string, turnTitle?: string) => void;
  onSelectCitation: (chunkId: string) => void;
}) {
  // Hydrate every stored response so unselected turns never fall back to raw text.
  const structuredAnswer = useEvidence(message.rag_run_id ?? undefined);
  if (!message.rag_run_id) {
    return (
      <article className="message message-assistant structured-unavailable">
        <div className="message-heading">
          <span className="message-role">Assistant</span>
          <CopyButton label="Copy answer" text={message.content} />
        </div>
        <p>
          {structuredAnswer.error && isNotFoundError(structuredAnswer.error)
            ? "This response is no longer available."
            : "Structured evidence could not be loaded for this response."}
        </p>
      </article>
    );
  }
  if (structuredAnswer.isLoading) {
    return (
      <article className="message message-assistant generation">
        <span className="message-role">Assistant</span>
        <span>Loading structured answer…</span>
      </article>
    );
  }
  if (structuredAnswer.error || !structuredAnswer.data) {
    return (
      <article className="message message-assistant structured-unavailable">
        <div className="message-heading">
          <span className="message-role">Assistant</span>
          <CopyButton label="Copy answer" text={message.content} />
        </div>
        <p>Structured evidence could not be loaded for this response.</p>
        <Button onClick={() => structuredAnswer.refetch()}>Retry</Button>
      </article>
    );
  }
  return (
    <AssistantAnswer
      activeChunk={activeChunk}
      answer={structuredAnswer.data}
      onSelectAnswer={onSelectAnswer}
      onSelectCitation={onSelectCitation}
    />
  );
}

function AssistantAnswer({
  answer,
  activeChunk,
  onSelectAnswer,
  onSelectCitation,
}: {
  answer: GroundedAnswer;
  activeChunk: string | null;
  onSelectAnswer: (runId: string, turnTitle?: string) => void;
  onSelectCitation: (chunkId: string) => void;
}) {
  // Render the current answer with evidence-backed citation controls.
  if (answer.insufficient_evidence) {
    return (
      <article
        className="message message-assistant insufficient"
        onClick={() => onSelectAnswer(answer.rag_run_id, answer.turn_title ?? undefined)}
      >
        <div className="message-heading">
          <span className="message-role">Assistant</span>
          <CopyButton label="Copy answer" text={answer.answer_text} />
        </div>
        <strong>Insufficient evidence in this project</strong>
        <p>{answer.answer_text}</p>
        <RetrievedChunkList answer={answer} onSelectCitation={onSelectCitation} />
      </article>
    );
  }

  return (
    <article
      className="message message-assistant structured-answer"
      onClick={() => onSelectAnswer(answer.rag_run_id, answer.turn_title ?? undefined)}
    >
      <div className="message-heading">
        <span className="message-role">Assistant</span>
        <CopyButton label="Copy answer" text={answer.answer_text} />
      </div>
      {answer.claims.map((claim) => (
        <p
          className={
            claim.citations.some(
              (citation) => citation.chunk_id === activeChunk,
            )
              ? "active-claim"
              : ""
          }
          key={claim.id}
        >
          {claim.text}{" "}
          <span className="citation-group">
            {claim.citations.map((citation) => {
              const evidence = answer.evidence[citation.chunk_id];
              return (
                evidence && (
                  <button
                    className={`citation ${
                      citation.chunk_id === activeChunk ? "citation-related" : ""
                    }`}
                    key={citation.chunk_id}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelectAnswer(
                        answer.rag_run_id,
                        answer.turn_title ?? undefined,
                      );
                      onSelectCitation(citation.chunk_id);
                    }}
                    title={evidence.excerpt}
                  >
                    <CitationBadgeLabel label={evidence.reference_label} />
                  </button>
                )
              );
            })}
          </span>
        </p>
      ))}
      <RetrievedChunkList answer={answer} onSelectCitation={onSelectCitation} />
    </article>
  );
}

function RetryButton({
  disabled,
  onRetry,
  prompt,
}: {
  disabled: boolean;
  onRetry: () => void;
  prompt: string;
}) {
  return (
    <button
      aria-label="Rerun prompt"
      className="message-icon-button"
      disabled={disabled}
      onClick={(event) => {
        event.stopPropagation();
        onRetry();
      }}
      title={`Rerun: ${prompt}`}
      type="button"
    >
      <span aria-hidden="true">↻</span>
    </button>
  );
}

function CopyButton({ label, text }: { label: string; text: string }) {
  const [copied, setCopied] = useState(false);

  const copy = async (): Promise<void> => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = text;
        textarea.setAttribute("readonly", "true");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        textarea.remove();
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  };

  return (
    <button
      aria-label={label}
      className="message-icon-button"
      onClick={(event) => {
        event.stopPropagation();
        void copy();
      }}
      title={label}
      type="button"
    >
      <span aria-hidden="true">{copied ? "✓" : "⧉"}</span>
    </button>
  );
}

function CitationBadgeLabel({ label }: { label: string }) {
  // Split only the display label's final chunk-order suffix; the canonical ID stays untouched.
  const match = /^(.*)_(\d+)$/.exec(label);
  if (!match) {
    return <span className="citation-document">{label}</span>;
  }
  return (
    <>
      <span className="citation-document">{match[1]}</span>
      <span className="citation-chunk">C{match[2]}</span>
    </>
  );
}

function RetrievedChunkList({
  answer,
  onSelectCitation,
}: {
  answer: GroundedAnswer;
  onSelectCitation: (chunkId: string) => void;
}) {
  // Use the same canonical claim citations that render beside the answer text.
  const retrieved = Object.values(answer.evidence);
  const citedChunkIds = new Set(
    answer.claims.flatMap((claim) =>
      claim.citations.map((citation) => citation.chunk_id),
    ),
  );
  return (
    <section className="retrieved-chunk-summary">
      <strong>Retrieved chunks</strong>
      <div className="retrieved-chunk-tags">
        {retrieved.map((evidence) => (
          <button
            className={`retrieved-chunk-tag ${citedChunkIds.has(evidence.chunk_id) ? "used" : ""}`}
            key={evidence.chunk_id}
            onClick={(event) => {
              event.stopPropagation();
              onSelectCitation(evidence.chunk_id);
            }}
            title={evidence.excerpt}
            type="button"
          >
            {evidence.reference_label}
          </button>
        ))}
      </div>
      <small>Highlighted chunks directly support the answer.</small>
    </section>
  );
}
