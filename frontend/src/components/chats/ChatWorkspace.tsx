import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";

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
  messages?: Message[];
  answer: GroundedAnswer | null;
  loading: boolean;
  error?: Error | null;
  activeChunk: string | null;
  onSelectAnswer: (runId: string, turnTitle?: string) => void;
  onSelectCitation: (chunkId: string) => void;
  onAsk: (message: string) => void;
};

/** Render a chat as navigable prompt-and-answer turns. */
export function ChatWorkspace({
  messages,
  answer,
  loading,
  error,
  activeChunk,
  onSelectAnswer,
  onSelectCitation,
  onAsk,
}: ChatWorkspaceProps) {
  const [value, setValue] = useState("");
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
  const showTurnNavigation = completedTurns.length > 1;

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
      return;
    }
    scrollToLatest();
  }, [loading, messages, messages?.length]);

  const send = (): void => {
    if (value.trim()) {
      onAsk(value.trim());
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
          {error && <ErrorState error={error} />}
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
                <MessageArticle message={turn.prompt} />
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
          disabled={loading}
          id="question"
          onChange={(event) => setValue(event.target.value)}
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
}: {
  message: Message;
}) {
  // Render a user prompt without treating conversation history as evidence.
  return (
    <article className={`message message-${message.role}`}>
      <span className="message-role">You</span>
      <div>{message.content}</div>
    </article>
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
        <span className="message-role">Assistant</span>
        <p>Structured evidence is unavailable for this older response.</p>
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
        <span className="message-role">Assistant</span>
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
        <span className="message-role">Assistant</span>
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
      <span className="message-role">Assistant</span>
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
                    className="citation"
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
                    {evidence.reference_label}
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
