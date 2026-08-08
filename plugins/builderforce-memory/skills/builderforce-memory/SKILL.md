---
name: builderforce-memory
description: Token-saving persistent memory. Recall durable facts (decisions, fixes, project context, user preferences) instead of re-reading files or chat history, and remember new durable facts as you learn them.
---

# builderforce-memory — the conductor

Persistent memory backed by the `builderforce-memory` MCP server. The point is
**token reduction**: pull a small relevant slice from memory instead of re-reading
the full file/history. The install also wires hooks so this is largely automatic:
the SessionStart digest, contextual recall on each prompt, and an autonomous Stop
capture that won't let a durable correction go unrecorded.

## Tools
- `memory_recall(query, [k])` — top-K relevant memories. First move on a task.
- `memory_get(key)` — exact lookup by key.
- `memory_recall_by_tag(tag)` — everything under a tag.
- `memory_remember(key, content, [tags], [importance])` — store ONE durable fact.
- `memory_forget(key)` — drop a superseded fact.

## Discipline
Reuse a stable key to REPLACE a fact (write-through), don't append duplicates.
Store decisions, non-obvious fixes, project constraints, user preferences — one
tight line each. Not things git already records or one-off turn details.
