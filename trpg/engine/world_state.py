from dataclasses import dataclass, field
from typing import Optional
from .character import Character, CombatState


def _present_in_current_room(ws) -> list[str]:
    """Snapshot of alive characters considered in the current scene.

    NPCs: those listed in current_room.npc_ids.
    PCs:  all non-NPC characters (party currently moves as one).
    """
    present: list[str] = []
    seen: set[str] = set()
    if ws.dungeon_map:
        present.extend(ws.dungeon_map.current_room.npc_ids)
    for cid, char in ws.characters.items():
        if not char.is_npc:
            present.append(cid)
    out: list[str] = []
    for cid in present:
        if cid in seen:
            continue
        seen.add(cid)
        c = ws.characters.get(cid)
        if c and c.is_alive():
            out.append(cid)
    return out


@dataclass
class WorldState:
    characters: dict          # str → Character (both PCs and NPCs)
    scene: str
    combat: Optional[CombatState] = None
    dungeon_map: Optional[object] = None   # DungeonMap — Optional import to avoid circular
    event_log: list = field(default_factory=list)   # legacy display log (string list)
    scenario_name: str = ""
    pending_conversation: str = ""  # npc_id to enter conversation with after GM narration
    quests: dict = field(default_factory=dict)   # str → Quest
    party_ids: list = field(default_factory=lambda: ["aria", "thor"])
    # Characters currently following the party. PCs are always in this list;
    # NPCs join via [RECRUIT] and leave on death / attack.
    narrative_log: list = field(default_factory=list)
    # Each entry: {"speaker": str, "text": str, "room_id": str|None, "present": list[str]}
    # speaker is a char_id, or "gm", or "system".

    def log_event(self, speaker: str, text: str, *,
                  room_id: Optional[str] = None,
                  present: Optional[list[str]] = None) -> dict:
        """Append an entry to narrative_log.

        Defaults: room_id = current room id; present = alive chars in current room.
        Returns the appended entry (mostly for tests).
        """
        if room_id is None and self.dungeon_map:
            room_id = self.dungeon_map.current_room_id
        if present is None:
            present = _present_in_current_room(self)
        entry = {
            "speaker": speaker,
            "text": text,
            "room_id": room_id,
            "present": list(present),
        }
        self.narrative_log.append(entry)
        return entry

    def log_for(self, char_id: str) -> list[dict]:
        """Entries the given character witnessed, in chronological order."""
        return [e for e in self.narrative_log if char_id in e["present"]]

    def log_all(self) -> list[dict]:
        """Full unfiltered log — used by GM (omniscient narrator)."""
        return list(self.narrative_log)

    def is_party_ally(self, char_id: str) -> bool:
        """True for PCs and NPCs currently following the party."""
        return char_id in self.party_ids

    def speaker_label(self, speaker: str) -> str:
        """Display label for a speaker key (for rendering log lines)."""
        if speaker == "gm":
            return "GM"
        if speaker == "system":
            return "系統"
        char = self.characters.get(speaker)
        return char.name if char else speaker
