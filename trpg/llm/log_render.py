"""Convert narrative_log entries into chat-completion messages.

Same log, different lenses: each agent sees its own utterances as `assistant`
and everyone else's as `user` with a `[speaker]：` prefix.

GM is the omniscient narrator — sees the unfiltered log via log_all().
Other identities use log_for(char_id) to get their personally-witnessed subset.
"""
from __future__ import annotations


def render_messages(world_state, identity: str) -> list[dict]:
    """Return chat messages for `identity`'s perspective.

    identity: "gm" (sees everything) or a char_id (sees only entries
              where present includes them).
    """
    entries = world_state.log_all() if identity == "gm" else world_state.log_for(identity)
    msgs: list[dict] = []
    for e in entries:
        if e["speaker"] == identity:
            msgs.append({"role": "assistant", "content": e["text"]})
        else:
            label = world_state.speaker_label(e["speaker"])
            msgs.append({"role": "user", "content": f"{label}：{e['text']}"})
    return msgs


def render_script(world_state, identity: str) -> str:
    """Plain-text script of the log from identity's POV (one line per entry).

    Used by NPC combat / social-DC agents that want a single user message
    instead of role-alternated messages.
    """
    entries = world_state.log_all() if identity == "gm" else world_state.log_for(identity)
    lines: list[str] = []
    for e in entries:
        label = world_state.speaker_label(e["speaker"])
        lines.append(f"{label}：{e['text']}")
    return "\n".join(lines)
