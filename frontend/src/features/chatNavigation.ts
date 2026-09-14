import type { Chat } from "../api/types";

function chatActivityTime(chat: Chat): number {
  const timestamp = Date.parse(chat.last_answer_at ?? chat.created_at);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

/** Sort chats by their latest answer, or creation time before their first answer. */
export function sortChatsByActivity(chats: Chat[]): Chat[] {
  return [...chats].sort(
    (left, right) =>
      chatActivityTime(right) - chatActivityTime(left) ||
      Date.parse(right.created_at) - Date.parse(left.created_at) ||
      left.id.localeCompare(right.id),
  );
}

/** Choose the remembered chat when it still exists, otherwise use the newest one. */
export function selectChatForWorkspace(
  chats: Chat[],
  lastVisitedChatId?: string,
): Chat | undefined {
  const rememberedChat = chats.find((chat) => chat.id === lastVisitedChatId);
  return rememberedChat ?? sortChatsByActivity(chats)[0];
}
