import { describe, expect, it } from "vitest";

import type { Chat } from "../api/types";
import { selectChatForWorkspace, sortChatsByActivity } from "./chatNavigation";

const chats: Chat[] = [
  {
    id: "unanswered",
    project_id: "project",
    name: "New chat",
    created_at: "2026-09-12T08:00:00Z",
    chat_name_set: false,
    last_answer_at: null,
  },
  {
    id: "older-answer",
    project_id: "project",
    name: "Earlier answer",
    created_at: "2026-09-10T08:00:00Z",
    chat_name_set: true,
    last_answer_at: "2026-09-11T08:00:00Z",
  },
  {
    id: "latest-answer",
    project_id: "project",
    name: "Latest answer",
    created_at: "2026-09-09T08:00:00Z",
    chat_name_set: true,
    last_answer_at: "2026-09-13T08:00:00Z",
  },
];

describe("chat workspace navigation", () => {
  it("orders chats by their latest completed answer", () => {
    expect(sortChatsByActivity(chats).map((chat) => chat.id)).toEqual([
      "latest-answer",
      "unanswered",
      "older-answer",
    ]);
  });

  it("returns the remembered chat before falling back to the newest chat", () => {
    expect(selectChatForWorkspace(chats, "older-answer")?.id).toBe(
      "older-answer",
    );
    expect(selectChatForWorkspace(chats, "removed")?.id).toBe("latest-answer");
  });
});
