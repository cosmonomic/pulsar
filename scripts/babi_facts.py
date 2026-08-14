"""State machine for locating the decisive fact in a BABILong qa2 record.

qa2 (bAbI task 2, "two supporting facts") tells a story about four actors
moving between rooms and picking up, dropping, and re-picking up items, then
asks where one item currently is. This module re-derives the answer from the
fact sentences alone and reports which single sentence fixes it.

Fact sentences are drawn from a fixed, closed vocabulary:

  actors: Daniel, John, Mary, Sandra
  items: apple, football, milk
  rooms: bathroom, bedroom, garden, hallway, kitchen, office
  movement verbs: went to, went back to, journeyed to, travelled to, moved to
  pickup verbs: took, got, grabbed, picked up
  drop verbs: dropped, left, discarded, put down

A movement sentence sets the actor's current room. A pickup sentence moves
the item into the actor's hand; while held, the item's room tracks the
holder's room as the holder moves. A drop sentence releases the item at the
actor's current room, where it stays until someone else picks it up.
"""

import json
import re
import statistics
from pathlib import Path

ACTORS = ["Daniel", "John", "Mary", "Sandra"]
ITEMS = ["apple", "football", "milk"]
ROOMS = ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"]

# Longest-first so "went back to" is tried before "went to".
MOVE_VERBS = ["went back to", "went to", "journeyed to", "travelled to", "moved to"]
PICKUP_VERBS = ["took", "got", "grabbed", "picked up"]
DROP_VERBS = ["dropped", "left", "discarded", "put down"]


def _alternation(words: list[str]) -> str:
    return "|".join(re.escape(word) for word in words)


_ACTOR_PATTERN = _alternation(ACTORS)
_ITEM_PATTERN = _alternation(ITEMS)
_ROOM_PATTERN = _alternation(ROOMS)
_MOVE_VERB_PATTERN = _alternation(MOVE_VERBS)
_PICKUP_VERB_PATTERN = _alternation(PICKUP_VERBS)
_DROP_VERB_PATTERN = _alternation(DROP_VERBS)

_FACT_SENTENCE = re.compile(
    r"\b(?:"
    rf"(?P<move_actor>{_ACTOR_PATTERN}) (?P<move_verb>{_MOVE_VERB_PATTERN}) the (?P<move_room>{_ROOM_PATTERN})"
    r"|"
    rf"(?P<pickup_actor>{_ACTOR_PATTERN}) (?P<pickup_verb>{_PICKUP_VERB_PATTERN}) the (?P<pickup_item>{_ITEM_PATTERN}) there"
    r"|"
    rf"(?P<drop_actor>{_ACTOR_PATTERN}) (?P<drop_verb>{_DROP_VERB_PATTERN}) the (?P<drop_item>{_ITEM_PATTERN})(?: there)?"
    r")\."
)


def babi_sentences(text: str) -> list[tuple[int, int, str]]:
    """Find every bAbI fact sentence in text, in order.

    Matches are found by the fixed actor/verb/item/room vocabulary, not by
    generic sentence splitting, so this works on both the clean bAbI text
    and a haystack with the same sentences embedded in filler.

    Returns:
        (char_lo, char_hi, sentence) tuples in the order they appear; the
        offsets are into text and sentence includes the trailing period.
    """
    return [
        (match.start(), match.end(), match.group(0))
        for match in _FACT_SENTENCE.finditer(text)
    ]


class _Event:
    """One parsed fact sentence.

    room is set only for "move" events; item only for "pickup" and "drop" events.
    """

    __slots__ = ("char_lo", "char_hi", "sentence", "kind", "actor", "room", "item")

    def __init__(
        self,
        char_lo: int,
        char_hi: int,
        sentence: str,
        kind: str,
        actor: str,
        room: str | None,
        item: str | None,
    ) -> None:
        self.char_lo = char_lo
        self.char_hi = char_hi
        self.sentence = sentence
        self.kind = kind
        self.actor = actor
        self.room = room
        self.item = item


def _parse_events(text: str) -> list[_Event]:
    events = []
    for match in _FACT_SENTENCE.finditer(text):
        if match.group("move_actor") is not None:
            event = _Event(
                match.start(),
                match.end(),
                match.group(0),
                "move",
                match.group("move_actor"),
                match.group("move_room"),
                None,
            )
        elif match.group("pickup_actor") is not None:
            event = _Event(
                match.start(),
                match.end(),
                match.group(0),
                "pickup",
                match.group("pickup_actor"),
                None,
                match.group("pickup_item"),
            )
        else:
            event = _Event(
                match.start(),
                match.end(),
                match.group(0),
                "drop",
                match.group("drop_actor"),
                None,
                match.group("drop_item"),
            )
        events.append(event)
    return events


_ITEM_IN_QUESTION = re.compile(_ITEM_PATTERN)


def _question_item(question: str) -> str | None:
    """None when zero or more than one item is named."""
    matches = _ITEM_IN_QUESTION.findall(question)
    if len(matches) != 1:
        return None
    return matches[0]


def _solve_qa2(
    events: list[_Event], item: str
) -> tuple[str | None, _Event | None, list[_Event]]:
    """Replay events and find the room, decisive event and evidence for one item.

    Returns:
        (room, decisive_event, evidence). The first two are None when the item is
        never mentioned or its holder never moved before the item was picked up.
        evidence is every sentence needed to reach the room, in source order: the
        decisive event plus the one that fixes its meaning. A drop states where the
        item was left only in combination with the actor's preceding move; a carried
        item's location is its holder's last move, which means nothing without the
        pickup that made them the holder.
    """
    actor_room: dict[str, str] = {}
    actor_room_event: dict[str, _Event] = {}
    holder: str | None = None
    holder_event: _Event | None = None
    resting_room: str | None = None
    resting_event: _Event | None = None
    resting_support: _Event | None = None

    for event in events:
        if event.kind == "move":
            actor_room[event.actor] = event.room
            actor_room_event[event.actor] = event
        elif event.item == item:
            if event.kind == "pickup":
                holder = event.actor
                holder_event = event
                resting_room = None
                resting_event = None
                resting_support = None
            elif event.kind == "drop":
                holder = None
                holder_event = None
                resting_room = actor_room.get(event.actor)
                resting_event = event
                resting_support = actor_room_event.get(event.actor)

    if holder is not None:
        decisive_event = actor_room_event.get(holder)
        if decisive_event is None:
            return None, None, []
        return (
            actor_room.get(holder),
            decisive_event,
            _in_source_order([holder_event, decisive_event]),
        )
    return (
        resting_room,
        resting_event,
        _in_source_order([resting_support, resting_event]),
    )


def _in_source_order(events: list[_Event | None]) -> list[_Event]:
    """The events that exist, deduplicated, ordered by position in the source."""
    seen: dict[int, _Event] = {}
    for event in events:
        if event is not None:
            seen[event.char_lo] = event
    return [seen[key] for key in sorted(seen)]


def decisive_span(record: dict, task: str = "qa2") -> tuple[int, int, str] | None:
    """Find the sentence that fixes the answer to a qa2 record's question.

    Returns:
        (char_lo, char_hi, sentence) of the decisive sentence inside
        record["input"], or None when the state machine cannot resolve it.
    """
    if task != "qa2":
        raise ValueError(f"unsupported task: {task}")
    item = _question_item(record["question"])
    if item is None:
        return None
    events = _parse_events(record["input"])
    _, decisive_event, _ = _solve_qa2(events, item)
    if decisive_event is None:
        return None
    return decisive_event.char_lo, decisive_event.char_hi, decisive_event.sentence


def evidence_spans(record: dict, task: str = "qa2") -> list[tuple[int, int, str]] | None:
    """Every sentence needed to answer a qa2 record's question, in source order.

    The decisive sentence alone does not fix the answer: a drop states where the item
    was left only together with the actor's preceding move, and a carried item's
    location is its holder's last move, which is meaningless without the pickup.

    Returns:
        [(char_lo, char_hi, sentence)] inside record["input"], ascending in char_lo,
        or None when the state machine cannot resolve the record.
    """
    if task != "qa2":
        raise ValueError(f"unsupported task: {task}")
    item = _question_item(record["question"])
    if item is None:
        return None
    events = _parse_events(record["input"])
    _, decisive_event, evidence = _solve_qa2(events, item)
    if decisive_event is None:
        return None
    return [(e.char_lo, e.char_hi, e.sentence) for e in evidence]


def predict(record: dict, task: str = "qa2") -> str | None:
    """The predicted room, or None when the state machine cannot resolve it."""
    if task != "qa2":
        raise ValueError(f"unsupported task: {task}")
    item = _question_item(record["question"])
    if item is None:
        return None
    events = _parse_events(record["input"])
    room, _, _ = _solve_qa2(events, item)
    return room


def _validate_split(
    split_name: str, split_path: Path, reference_sequences: list[list[str]] | None
) -> list[list[str]]:
    """Run predict and decisive_span over one split and print a summary.

    Returns:
        The bAbI sentence sequence (sentence text only) for every record in
        this split, for cross-checking against another split.
    """
    records = json.loads(split_path.read_text())
    correct = 0
    resolvable = 0
    sequence_matches = 0
    normalized_positions = []
    sequences = []

    for index, record in enumerate(records):
        sequence = [sentence for _, _, sentence in babi_sentences(record["input"])]
        sequences.append(sequence)
        if reference_sequences is not None and sequence == reference_sequences[index]:
            sequence_matches += 1

        if predict(record) == record["target"]:
            correct += 1

        span = decisive_span(record)
        if span is not None:
            resolvable += 1
            char_lo, _, _ = span
            normalized_positions.append(char_lo / len(record["input"]))

    print(f"{split_name}: predict correct {correct}/{len(records)}")
    print(f"{split_name}: decisive_span resolvable {resolvable}/{len(records)}")
    if reference_sequences is not None:
        print(
            f"{split_name}: sentence sequence matches 0k {sequence_matches}/{len(records)}"
        )
    if normalized_positions:
        median_position = statistics.median(normalized_positions)
        print(
            f"{split_name}: median normalized decisive position {median_position:.4f}"
        )
    return sequences


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data" / "babilong" / "qa2"
    zero_k_sequences = _validate_split("0k", data_dir / "0k.json", None)
    _validate_split("64k", data_dir / "64k.json", zero_k_sequences)
