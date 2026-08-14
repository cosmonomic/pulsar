"""BABILong long-context QA harness for the C++ engine.

Drives the Engine surface (create / feed / step / stats / release) over BABILong
records: bAbI reasoning facts hidden in a book-length haystack, shipped at fixed
context lengths from 0k to 1M. The task is held fixed and the input length varied,
so the result is a curve over length splits rather than a single number.

Two memory regimes:

  FULL:   recall off; the active buffer is sized to the whole prompt. A length split
          whose prompt exceeds the model's trained window is skipped
          ("over_native_window"), so FULL measures the model inside its native window
          and nothing else. A prompt that also exceeds --max_pages is skipped instead
          as "too_long_for_full".
  RECALL: recall on with a bounded active buffer (--active_buffer_size tokens).
          Eviction demotes over-window KV during chunked prefill; at generate-entry
          the engine recalls page-aligned segments into the freed headroom.

The system turn carries BABILong's task instruction, in-context examples and answer
format, and is the protected sink (n_sink tokens passed to create), so eviction never
touches them. The question follows the haystack, inside the working window.

Grading is BABILong's own closed-label metric: the reply is correct when the task's
label set, restricted to the reply and minus any label the question already names,
is exactly the target.

Engine plumbing (config dict, buffer sizing, stats readout, distribution probes) is
imported from eval_longbench so both harnesses drive the engine identically.
"""

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

import babi_facts
from eval_longbench import (
    META_PROMPTS,
    PAGE_SIZE,
    _sum_hists,
    _with_stats,
    build_engine,
    ceil_pages,
    derive_engine_args,
    format_skips,
    hist_median,
    report_density_by_position,
    report_histograms,
    report_skips,
    skip_counts,
    staging_tokens,
)
from pulsar import checkpoint
from pulsar.checkpoint import PulsarModel, PulsarModelConfig, Tokenizer
from pulsar.conversation import ChatMLConversation
from pulsar.rt import Session

BASE_URL = "https://huggingface.co/datasets/RMT-team/babilong/resolve/main/data"

# Splits published for qa1-qa5. 0k is the haystack-free bAbI original.
LENGTHS = [
    "0k",
    "1k",
    "2k",
    "4k",
    "8k",
    "16k",
    "32k",
    "64k",
    "128k",
    "256k",
    "512k",
    "1M",
]
TASKS = ["qa1", "qa2", "qa3", "qa4", "qa5"]

# Task prompt bodies, verbatim from RMT-team/babilong (babilong/prompts.py). A run
# that edits them is not comparable with a run that does not.
TASK_PROMPTS = {
    "qa1": {
        "instruction": (
            "I will give you context with the facts about positions of different "
            "persons hidden in some random text and a question. You need to answer "
            "the question based only on the information from the facts. If a person "
            "was in different locations, use the latest location to answer the "
            "question."
        ),
        "examples": (
            "<example>\n"
            "Charlie went to the hallway. Judith come back to the kitchen. Charlie "
            "travelled to balcony. Where is Charlie?\n"
            "Answer: The most recent location of Charlie is balcony.\n"
            "</example>\n\n"
            "<example>\n"
            "Alan moved to the garage. Charlie went to the beach. Alan went to the "
            "shop. Rouse travelled to balcony. Where is Alan?\n"
            "Answer: The most recent location of Alan is shop.\n"
            "</example>"
        ),
        "post_prompt": (
            "Always return your answer in the following format: The most recent "
            "location of ’person’ is ’location’. Do not write "
            "anything else after that."
        ),
    },
    "qa2": {
        "instruction": (
            "I give you context with the facts about locations and actions of "
            "different persons hidden in some random text and a question."
            "You need to answer the question based only on the information from the "
            "facts.\n"
            "If a person got an item in the first location and travelled to the "
            "second location the item is also in the second location. "
            "If a person dropped an item in the first location and moved to the "
            "second location the item remains in the first location."
        ),
        "examples": (
            "<example>\n"
            "Charlie went to the kitchen. Charlie got a bottle. Charlie moved to the "
            "balcony. Where is the bottle?\n"
            "Answer: The bottle is in the balcony.\n"
            "</example>\n"
            "<example>\n"
            "Alan moved to the garage. Alan got a screw driver. Alan moved to the "
            "kitchen. Where is the screw driver?\n"
            "Answer: The screw driver is in the kitchen.\n"
            "</example>"
        ),
        "post_prompt": (
            "Always return your answer in the following format: The ’item’ is in "
            "’location’. Do not write anything else after that."
        ),
    },
    "qa3": {
        "instruction": (
            "I give you context with the facts about locations and actions of "
            "different persons hidden in some random text and a question. "
            "You need to answer the question based only on the information from the "
            "facts.\n"
            "If a person got an item in the first location and travelled to the "
            "second location the item is also in the second location. "
            "If a person dropped an item in the first location and moved to the "
            "second location the item remains in the first location."
        ),
        "examples": (
            "<example>\n"
            "John journeyed to the bedroom. Mary grabbed the apple. Mary went back to "
            "the bathroom. Daniel journeyed to the bedroom. Daniel moved to the "
            "garden. Mary travelled to the kitchen. Where was the apple before the "
            "kitchen?\n"
            "Answer: Before the kitchen the apple was in the bathroom.\n"
            "</example>\n"
            "<example>\n"
            "John went back to the bedroom. John went back to the garden. John went "
            "back to the kitchen. Sandra took the football. Sandra travelled to the "
            "garden. Sandra journeyed to the bedroom. Where was the football before "
            "the bedroom?\n"
            "Answer: Before the bedroom the football was in the garden.\n"
            "</example>"
        ),
        "post_prompt": (
            "Always return your answer in the following format: Before the "
            "$location_1$ the $item$ was in the $location_2$. Do not write anything "
            "else after that."
        ),
    },
    "qa4": {
        "instruction": (
            "I will give you context with the facts about different people, their "
            "location and actions, hidden in some random text and a question. "
            "You need to answer the question based only on the information from the "
            "facts."
        ),
        "examples": (
            "<example>\n"
            "The hallway is south of the kitchen. The bedroom is north of the "
            "kitchen. What is the kitchen south of?\n"
            "Answer: bedroom\n"
            "</example>\n"
            "<example>\n"
            "The garden is west of the bedroom. The bedroom is west of the kitchen. "
            "What is west of the bedroom?\n"
            "Answer: garden\n"
            "</example>"
        ),
        "post_prompt": (
            "Your answer should contain only one word - location. Do not write "
            "anything else after that."
        ),
    },
    "qa5": {
        "instruction": (
            "I will give you context with the facts about locations and their "
            "relations hidden in some random text and a question. You need to answer "
            "the question based only on the information from the facts."
        ),
        "examples": (
            "<example>\n"
            "Mary picked up the apple there. Mary gave the apple to Fred. Mary moved "
            "to the bedroom. Bill took the milk there. Who did Mary give the apple "
            "to?\n"
            "Answer: Fred\n"
            "</example>\n"
            "<example>\n"
            "Jeff took the football there. Jeff passed the football to Fred. Jeff got "
            "the milk there. Bill travelled to the bedroom. Who gave the football?\n"
            "Answer: Jeff\n"
            "</example>\n"
            "<example>\n"
            "Fred picked up the apple there. Fred handed the apple to Bill. Bill "
            "journeyed to the bedroom. Jeff went back to the garden. What did Fred "
            "give to Bill?\n"
            "Answer: apple\n"
            "</example>"
        ),
        "post_prompt": (
            "Your answer should contain only one word. Do not write anything else "
            "after that. Do not explain your answer."
        ),
    },
}

# Closed answer set per task, verbatim from babilong/metrics.py. A uniform guess
# scores 1/len(labels), so chance is 0.167 for qa1-qa4 and 0.143 for qa5.
TASK_LABELS = {
    "qa1": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa2": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa3": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa4": ["bathroom", "bedroom", "garden", "hallway", "kitchen", "office"],
    "qa5": ["Bill", "Fred", "Jeff", "Mary", "apple", "football", "milk"],
}

# Qwen3 renders a disabled thinking block as an empty one right after the assistant
# turn opens; without it the model opens <think> itself.
EMPTY_THINK = "<think>\n\n</think>\n\n"

SYSTEM_TEMPLATE = "{instruction}\n\n{examples}\n\n{post_prompt}"
USER_TEMPLATE = "<context>\n{context}\n</context>\n\nQuestion: {question}"


def split_path(data_dir: Path, task: str, length: str) -> Path:
    return data_dir / task / f"{length}.json"


def download_split(data_dir: Path, task: str, length: str) -> Path:
    """Fetch data/<task>/<length>.json to the cache if absent. Exit on failure."""
    path = split_path(data_dir, task, length)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{task}/{length}.json"
    print(f"downloading {task}/{length} -> {path}")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # noqa: BLE001 - report and stop, do not fake data
        path.unlink(missing_ok=True)
        print(f"download failed: {exc}")
        print(f"fetch manually from {url} to {path}")
        sys.exit(1)
    return path


def load_split(data_dir: Path, task: str, length: str, limit: int) -> list[dict]:
    """First limit records of a split. Record index is stable across length splits."""
    records = json.loads(download_split(data_dir, task, length).read_text())
    return records[:limit] if limit else records


def build_system_text(task: str, meta: str) -> str:
    """BABILong's instruction, examples and answer format, plus the meta clause."""
    return SYSTEM_TEMPLATE.format(**TASK_PROMPTS[task]) + meta


def build_user_text(rec: dict) -> str:
    """Haystack in a <context> block with the question after it."""
    return USER_TEMPLATE.format(
        context=str(rec["input"]).strip(), question=str(rec["question"]).strip()
    )


def encode_prompt(
    tok: Tokenizer,
    prompter: ChatMLConversation,
    task: str,
    rec: dict,
    meta: str,
    think: bool,
) -> tuple[list[int], list[int]]:
    """ChatML system + user turns plus the open assistant turn, as token ids.

    Returns (system_ids, body_ids): lengths for sizing the active/recall buffers.
    The full stream is encoded once and split at the system length, so
    tokenization is identical to a single encode.
    """
    system_text = (
        prompter.turn_open("system")
        + build_system_text(task, meta)
        + prompter.turn_close()
    )
    n_sink = len(tok.encode(system_text).ids)
    text = (
        system_text
        + prompter.turn_open("user")
        + build_user_text(rec)
        + prompter.turn_close()
        + prompter.turn_open("assistant")
        + ("" if think else EMPTY_THINK)
    )
    full = tok.encode(text).ids
    return full[:n_sink], full[n_sink:]


# Tokens each evidence range is extended by below its sentence's first character,
# covering the prefix-tokenization boundary (see supporting_spans).
SUPPORT_ADDR_PAD = 2


def supporting_spans(
    tok: Tokenizer, prompter: ChatMLConversation, task: str, rec: dict, meta: str
) -> tuple[list[tuple[int, int]], str] | None:
    """The record's evidence as conversation address ranges [lo, hi), and its text.

    The addresses come from encoding the prompt truncated at each sentence's character
    bounds. A truncated prefix does not tokenize identically to the same prefix of the
    whole text, which costs a token at the low end, so each low bound is padded by
    SUPPORT_ADDR_PAD to keep the sentence's first word inside its range.
    Returns None for a task with no state machine, or when it cannot resolve the record.
    """
    if task != "qa2":
        return None
    context = str(rec["input"]).strip()
    spans = babi_facts.evidence_spans({**rec, "input": context}, task)
    if not spans:
        return None
    sentence = " ".join(text for _, _, text in spans)
    head = (
        prompter.turn_open("system")
        + build_system_text(task, meta)
        + prompter.turn_close()
        + prompter.turn_open("user")
        + USER_TEMPLATE.split("{context}")[0]
    )
    char_spans = [(lo, hi) for lo, hi, _ in spans]
    ranges = [
        (
            max(0, len(tok.encode(head + context[:lo]).ids) - SUPPORT_ADDR_PAD),
            len(tok.encode(head + context[:hi]).ids),
        )
        for lo, hi in char_spans
    ]
    return ranges, sentence


def address_pages(ranges: list[tuple[int, int]]) -> set[int]:
    """The page indices conversation address ranges [lo, hi) cover.

    A page is the eviction quantum, so a range straddling a page boundary covers both
    pages. An empty or inverted range covers none.
    """
    return {
        page
        for lo, hi in ranges
        if hi > lo
        for page in range(lo // PAGE_SIZE, (hi - 1) // PAGE_SIZE + 1)
    }


async def context_page_ids(session: Session) -> list[int]:
    """The attended pages' indices in address order, empty when there is no view."""
    try:
        return list((await session.get_context()).keys)
    except Exception:
        return []


def residency_label(resident: int, total: int) -> str:
    """How much of a page set the attention window holds; "none" when it is empty."""
    if total == 0:
        return "none"
    if resident == total:
        return "full"
    return "partial" if resident else "absent"


def answer_text(text: str) -> str:
    """The reply's answer region: after a closed thinking block, first sentence.

    Mirrors babilong's preprocess_output, with the thinking block stripped first so a
    sentence of reasoning cannot be read as the answer.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    text = text.lower().split(".")[0]
    for marker in ("<context>", "<example>", "question"):
        text = text.split(marker)[0]
    return text.strip()


def extract_labels(text: str, question: str, task: str) -> list[str]:
    """Task labels the reply commits to, minus those the question already names.

    A label the question mentions is never the target, so it cannot count as a
    prediction.
    """
    labels = {label.lower() for label in TASK_LABELS[task]}
    in_question = {label for label in labels if label in question.lower()}
    return sorted({label for label in labels if label in text} - in_question)


def grade(pred_labels: list[str], target: str) -> bool:
    """True when the reply names the target and nothing else from the label set."""
    return set(pred_labels) == {
        t.strip() for t in target.lower().split(",") if t.strip()
    }


def native_window(args: argparse.Namespace, config: PulsarModelConfig) -> int:
    """Trained context of the model in TOKENS.

    A returned 0 means the model config declares no window, and disables the FULL-mode
    skip. --native_window 0 is "read the config", not "disable".
    """
    return args.native_window or config.max_position_embeddings


@dataclass
class SampleResult:
    id: str
    task: str
    length_split: str
    mode: str
    gold: str
    pred: str | None
    correct: bool
    prompt_tokens: int
    gen_tokens: int
    demoted: int
    skip_reason: str | None
    raw: str | None = None
    active_pages: int = 0  # attended pages at end of generation (recall mode)
    evicted_pages: int = 0  # PAGES demoted over the run (recall mode)
    # Split of evicted_pages by cause: pages the keep-bar took, and pages the
    # active_max_size backstop took on top of them. Each _decode field is the
    # decode-cadence subset of the plainly named one, so prefill is the difference.
    bar_evicted_pages: int = 0
    bar_evicted_pages_decode: int = 0
    backstop_evicted_pages: int = 0
    backstop_evicted_pages_decode: int = 0
    # Age-correction counterfactual: of raw_mass_victim_pages victims, how many a
    # same-sized selection over the UNCORRECTED mass also takes.
    raw_mass_victim_pages: int = 0
    raw_mass_victim_pages_decode: int = 0
    raw_mass_agree_pages: int = 0
    raw_mass_agree_pages_decode: int = 0
    recalled_pages: int = 0  # PAGES recalled over the run (recall mode)
    cycles: int = 0  # evict->recall cycles that fired (recall mode)
    decode_cycles: int = 0
    # Active length in TOKENS summed over cycles at the two points a cycle moves it:
    # after eviction and after recall. cycles divides either into a mean.
    post_evict_tokens: int = 0
    post_evict_tokens_decode: int = 0
    post_recall_tokens: int = 0
    post_recall_tokens_decode: int = 0
    # Cycles whose eviction freed nothing; recall returned before scoring.
    starved_cycles: int = 0
    starved_cycles_decode: int = 0
    staged_hits: int = 0  # candidates already resident in the staging cache (qk only)
    staged_misses: int = 0  # candidates read out of a bucket (qk only)
    promoted_s1: int = 0  # PAGES promoted out of the RAM bucket
    promoted_s2: int = 0  # PAGES promoted out of the disk bucket
    density_hist: list[int] = field(default_factory=list)
    density_log10_lo: int = 0
    density_log10_hi: int = 0
    # Page-age histograms in TOKENS (stream_len minus the page's base address), log10
    # bins over [10**age_log10_lo, 10**age_log10_hi]: the demoted pages against every
    # evictable page, so the two compare.
    evicted_age_hist: list[int] = field(default_factory=list)
    evicted_age_hist_decode: list[int] = field(default_factory=list)
    candidate_age_hist: list[int] = field(default_factory=list)
    candidate_age_hist_decode: list[int] = field(default_factory=list)
    age_log10_lo: int = 0
    age_log10_hi: int = 0
    recall_hist: list[int] = field(default_factory=list)
    recall_hist_decode: list[int] = field(default_factory=list)
    recall_lo: int = 0
    recall_hi: int = 0
    position_raw_mass_sum: list[float] = field(default_factory=list)
    position_corrected_density_sum: list[float] = field(default_factory=list)
    position_page_count: list[int] = field(default_factory=list)
    # The record's evidence and whether the attention window still held it. The ranges
    # are conversation addresses [lo, hi); addr_lo/addr_hi are their envelope.
    support_sentence: str | None = None
    support_ranges: list[list[int]] = field(default_factory=list)
    support_addr_lo: int = -1
    support_addr_hi: int = -1
    support_inside_working: bool = False  # inside the protected window at prefill end
    # Evidence pages attended at the two sampled points, out of support_pages. The
    # labels are those counts as full/partial/absent (see residency_label).
    support_pages: int = 0
    support_resident_prefill: int = 0
    support_resident_end: int = 0
    support_residency_prefill: str = "none"
    support_residency_end: str = "none"


def _plan_buffers(
    mode: str, n_prompt: int, args: argparse.Namespace
) -> tuple[int, int] | str:
    """(active_buffer_size, recall_buffer_size) in TOKENS, or a skip reason.

    FULL keeps the whole prompt plus generation attended and refuses anything past the
    trained window; RECALL sizes the buffer to the bounded window and the
    evict->recall cycle works to active_buffer_size - max_chunk_size.
    """
    if mode == "full":
        need = n_prompt + args.gen_tokens
        if args.native_window_tokens and need > args.native_window_tokens:
            return "over_native_window"
        need_pages = ceil_pages(need) + 2
        if need_pages > args.max_pages:
            return "too_long_for_full"
        return need_pages * PAGE_SIZE, 0
    return ceil_pages(args.active_buffer_size) * PAGE_SIZE, staging_tokens(args)


def run_sample(
    loaded: PulsarModel,
    prompter: ChatMLConversation,
    task: str,
    length: str,
    index: int,
    rec: dict,
    mode: str,
    args: argparse.Namespace,
) -> SampleResult:
    system_ids, body_ids = encode_prompt(
        loaded.tokenizer,
        prompter,
        task,
        rec,
        META_PROMPTS[args.meta_prompt],
        args.think,
    )
    n_prompt = len(system_ids) + len(body_ids)
    gold = str(rec["target"]).strip()
    rid = f"{task}/{index}"

    def skip(reason: str) -> SampleResult:
        return SampleResult(
            id=rid,
            task=task,
            length_split=length,
            mode=mode,
            gold=gold,
            pred=None,
            correct=False,
            prompt_tokens=n_prompt,
            gen_tokens=0,
            demoted=0,
            skip_reason=reason,
        )

    plan = _plan_buffers(mode, n_prompt, args)
    if isinstance(plan, str):
        return skip(plan)
    active_buffer_size, recall_buffer_size = plan

    support = (
        supporting_spans(
            loaded.tokenizer, prompter, task, rec, META_PROMPTS[args.meta_prompt]
        )
        if mode == "recall"
        else None
    )
    support_ranges, support_sentence = support if support else ([], None)
    support_pages = address_pages(support_ranges)

    async def _run() -> tuple[list[int], int, dict, int, int, int]:
        pulsar = build_engine(loaded, args, active_buffer_size, recall_buffer_size)
        try:
            session = await pulsar.create_session(system_ids)
            # send() only queues the feed; the first token off the response lands
            # the whole prefill, so this is where the prompt's residency can first
            # be read.
            response = await session.send(body_ids, args.gen_tokens)
            stream = aiter(response)
            first = await anext(stream, None)
            resident_prefill = (
                len(support_pages & set(await context_page_ids(session)))
                if support_pages
                else 0
            )
            out = [first] if first is not None else []
            async for tok in stream:
                out.append(tok)
            demoted = await session.get_demoted_tokens_count()
            st = await session.get_stats()
            end_page_ids = await context_page_ids(session)
            active_pages = len(end_page_ids)
            resident_end = len(support_pages & set(end_page_ids))
            await session.destroy()
            return out, demoted, st, active_pages, resident_prefill, resident_end
        finally:
            await pulsar.aclose()

    try:
        out, demoted, st, active_pages, resident_prefill, resident_end = asyncio.run(
            _run()
        )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return skip("OOM")

    decoded = loaded.tokenizer.decode(out)
    labels = extract_labels(answer_text(decoded), str(rec["question"]), task)
    return _with_stats(
        SampleResult(
            id=rid,
            task=task,
            length_split=length,
            mode=mode,
            gold=gold,
            pred=",".join(labels) if labels else None,
            correct=grade(labels, gold),
            prompt_tokens=n_prompt,
            gen_tokens=len(out),
            demoted=demoted,
            active_pages=active_pages,
            skip_reason=None,
            raw=decoded,
            support_sentence=support_sentence,
            support_ranges=[[lo, hi] for lo, hi in support_ranges],
            support_addr_lo=min((lo for lo, _ in support_ranges), default=-1),
            support_addr_hi=max((hi for _, hi in support_ranges), default=-1),
            support_inside_working=bool(support_ranges)
            and n_prompt - min(lo for lo, _ in support_ranges) <= args.n_working,
            support_pages=len(support_pages),
            support_resident_prefill=resident_prefill,
            support_resident_end=resident_end,
            support_residency_prefill=residency_label(
                resident_prefill, len(support_pages)
            ),
            support_residency_end=residency_label(resident_end, len(support_pages)),
        ),
        st,
    )


def summarize(results: list[SampleResult], mode: str, args: argparse.Namespace) -> None:
    """The length curve for one mode, plus the recall diagnostics.

    Occupancy and churn are diagnostics: accuracy against the native-window ceiling is
    the result. A split's skips are counted by reason on its own row, because a curve
    is read across splits and a thinned split cannot be told from a scored one by its
    accuracy alone. cap counts replies that ran into --gen_tokens: a truncated reply
    grades as wrong and is not one.
    """
    rows = [r for r in results if r.mode == mode]
    if not rows:
        return
    scored = [r for r in rows if r.skip_reason is None]
    skips = skip_counts(rows)
    print(f"\n=== mode={mode} task={args.task} ===")
    print(f"n_scored={len(scored)} of {len(rows)}")
    print(f"n_skipped={sum(skips.values())}: {format_skips(skips)}")
    print(f"n_hit_gen_cap={sum(r.gen_tokens >= args.gen_tokens for r in scored)}")

    by_len: dict[str, list[SampleResult]] = defaultdict(list)
    for r in rows:
        by_len[r.length_split].append(r)
    print(
        f"  {'length':>8} {'n':>4} {'acc':>6} {'no_ans':>7} {'cap':>4} {'tokens':>8} "
        f"{'gen':>5} {'skipped':>8}  reasons"
    )
    for length in args.lengths_list:
        group = by_len.get(length, [])
        if not group:
            continue
        ok = [r for r in group if r.skip_reason is None]
        split_skips = skip_counts(group)
        acc = sum(r.correct for r in ok) / len(ok) if ok else float("nan")
        mean_tokens = sum(r.prompt_tokens for r in group) / len(group)
        mean_gen = sum(r.gen_tokens for r in ok) / len(ok) if ok else 0.0
        print(
            f"  {length:>8} {len(ok):>4} {acc:>6.3f} "
            f"{sum(r.pred is None for r in ok):>7} "
            f"{sum(r.gen_tokens >= args.gen_tokens for r in ok):>4} "
            f"{mean_tokens:>8.0f} {mean_gen:>5.0f} "
            f"{sum(split_skips.values()):>8}  {format_skips(split_skips)}"
        )

    if mode == "recall" and scored:
        cand = sum(r.staged_hits + r.staged_misses for r in scored)
        promoted = sum(r.promoted_s1 + r.promoted_s2 for r in scored)
        hit_rate = sum(r.staged_hits for r in scored) / cand if cand else 0.0
        s2_share = sum(r.promoted_s2 for r in scored) / promoted if promoted else 0.0
        print(f"diagnostics: total_demoted_kv={sum(r.demoted for r in scored)}")
        print(f"  staging_hit_rate={hit_rate:.3f} over {cand} candidates")
        print(f"  s2_share_of_promotions={s2_share:.3f} over {promoted} promotions")
        print(
            f"  cycles={sum(r.cycles for r in scored)} "
            f"evicted_pages={sum(r.evicted_pages for r in scored)} "
            f"recalled_pages={sum(r.recalled_pages for r in scored)}"
        )
        report_evict_attribution(scored)


def report_evict_attribution(scored: list[SampleResult]) -> None:
    """What caused the run's evictions, prefill cycles against decode cycles.

    Every engine counter carries the run total and its decode subset, so the prefill
    half is the difference. bar/backstop attribute a demotion to the keep-bar or to the
    buffer running out of room; raw_mass_agreement is the share of victims a same-sized
    selection over the UNCORRECTED mass also takes, so 1.0 means the age correction did
    not move the decision; the median ages say what the selection could take
    (candidate) against what it took (evicted), which the histograms in the JSON carry
    in full.
    """
    if not scored:
        return

    def counters(name: str) -> tuple[int, int]:
        total = sum(getattr(r, name) for r in scored)
        decode = sum(getattr(r, f"{name}_decode") for r in scored)
        return total - decode, decode

    def median_ages(name: str) -> tuple[float, float]:
        total = _sum_hists([getattr(r, name) for r in scored])
        decode = _sum_hists([getattr(r, f"{name}_decode") for r in scored])
        prefill = [t - d for t, d in zip(total, decode, strict=True)]
        lo, hi = scored[0].age_log10_lo, scored[0].age_log10_hi
        return hist_median(prefill, lo, hi), hist_median(decode, lo, hi)

    def share(agree: int, victims: int) -> float:
        return agree / victims if victims else float("nan")

    bar = counters("bar_evicted_pages")
    backstop = counters("backstop_evicted_pages")
    victims = counters("raw_mass_victim_pages")
    agree = counters("raw_mass_agree_pages")
    evicted_age = median_ages("evicted_age_hist")
    candidate_age = median_ages("candidate_age_hist")
    for phase, i in (("prefill", 0), ("decode", 1)):
        print(
            f"  {phase:>7}: bar_evicted_pages={bar[i]} "
            f"backstop_evicted_pages={backstop[i]} "
            f"raw_mass_agreement={share(agree[i], victims[i]):.3f} "
            f"over {victims[i]} victims"
        )
        print(
            f"           median_age_tokens evicted={evicted_age[i]:.3g} "
            f"candidate={candidate_age[i]:.3g}"
        )


def report_support_residency(results: list[SampleResult]) -> None:
    """Whether the attention window still held the record's evidence, and accuracy.

    Sampled at two points the harness controls: when the prefill lands, which is the
    residency generation starts from, and when generation ends. The split by n_working
    separates evidence the protected window keeps by construction from evidence that
    had to survive eviction.
    """
    checked = [r for r in results if r.support_pages and r.skip_reason is None]
    if not checked:
        return
    print("\n=== evidence residency in the attention window ===")
    for group_name, group in (
        ("inside n_working", [r for r in checked if r.support_inside_working]),
        ("outside n_working", [r for r in checked if not r.support_inside_working]),
    ):
        if not group:
            continue
        acc = sum(r.correct for r in group) / len(group)
        print(f"\n  {group_name}: n={len(group)} acc={acc:.3f}")
        pages = sum(r.support_pages for r in group)
        for title, label_field, resident_field in (
            ("at prefill end", "support_residency_prefill", "support_resident_prefill"),
            ("at generation end", "support_residency_end", "support_resident_end"),
        ):
            resident = sum(getattr(r, resident_field) for r in group)
            labels = Counter(getattr(r, label_field) for r in group)
            print(f"    {title} ({resident}/{pages} evidence pages attended):")
            for label, count in labels.most_common():
                held = [r for r in group if getattr(r, label_field) == label]
                acc_held = sum(r.correct for r in held) / len(held)
                print(
                    f"      {label:>8} {count:>4} ({count / len(group):.3f})"
                    f"  acc={acc_held:.3f}"
                )


def report_curve(results: list[SampleResult], args: argparse.Namespace) -> None:
    """Accuracy by length split with the modes side by side.

    Each cell carries n as scored/attempted, so a point standing on fewer records than
    its neighbours reads off the curve rather than out of the skip report.
    """
    modes = sorted({r.mode for r in results})
    print(f"\n=== accuracy by length split (task={args.task}) ===")
    print("  " + f"{'length':>8}" + "".join(f"{m:>20}" for m in modes))
    for length in args.lengths_list:
        cells = []
        for mode in modes:
            group = [r for r in results if r.mode == mode and r.length_split == length]
            ok = [r for r in group if r.skip_reason is None]
            if ok:
                acc = sum(r.correct for r in ok) / len(ok)
                cell = f"{acc:.3f} n={len(ok)}/{len(group)}"
            else:
                reasons = skip_counts(group)
                cell = next(iter(reasons)) if len(reasons) == 1 else "skipped"
            cells.append(cell.rjust(20))
        print(f"  {length:>8}" + "".join(cells))


def dump_raw(results: list[SampleResult], dump_dir: Path, per_mode: int = 3) -> None:
    """Write a few raw replies per mode to dump_dir for inspection."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / "raw_babilong_samples.txt"
    lines: list[str] = []
    for mode in sorted({r.mode for r in results}):
        picked = [r for r in results if r.mode == mode and r.raw is not None][:per_mode]
        for r in picked:
            lines.append(
                f"===== mode={mode} id={r.id} len={r.length_split} "
                f"gold={r.gold} pred={r.pred} ====="
            )
            lines.append(r.raw or "")
            lines.append("")
    if lines:
        out.write_text("\n".join(lines))
        print(f"\nwrote raw samples -> {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="qwen3-8b")
    p.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/babilong"),
        help="split cache; a missing <task>/<length>.json is downloaded into it",
    )
    p.add_argument("--task", choices=TASKS, default="qa2")
    p.add_argument(
        "--lengths",
        type=str,
        default="4k,8k,16k,32k",
        help=f"comma-separated length splits to run, from {','.join(LENGTHS)}. The same "
        "record indices are run at every split, so the result is a curve over length "
        "with the questions held fixed",
    )
    p.add_argument(
        "--limit", type=int, default=20, help="records per split; 0 runs all 100"
    )
    p.add_argument(
        "--meta_prompt",
        choices=list(META_PROMPTS),
        default="none",
        help="metacognition clause appended to the system prompt",
    )
    p.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="let the model open its own <think> block; off injects an empty one, which "
        "is what BABILong's answer format asks for. Raise --gen_tokens when on",
    )
    p.add_argument("--mode", choices=["full", "recall", "both"], default="full")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--gen_tokens", type=int, default=64)
    p.add_argument(
        "--native_window",
        type=int,
        default=0,
        help="trained context in TOKENS that FULL refuses to exceed; 0 reads the "
        "model's max_position_embeddings. FULL past this is not a baseline",
    )
    # Decode sampler. Distinct from --recall_temperature, which softens the recall
    # keep-bar and does not touch decoding.
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="decode sampling temperature; 0 => greedy (argmax)",
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="nucleus sampling mass; 1.0 => no truncation. Only applies when "
        "temperature > 0",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="truncate to the k highest-probability tokens; 0 => no limit. Only applies "
        "when temperature > 0",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="sampler RNG seed; -1 draws a random one, which makes a sampled run "
        "unreproducible. Also seeds the recall selection draw",
    )
    p.add_argument(
        "--max_pages",
        type=int,
        default=3000,
        help="KV pool ceiling in pages. A FULL prompt past this is skipped as "
        "too_long_for_full, on top of the over_native_window skip; a recall-mode "
        "buffer over it is a startup error",
    )
    p.add_argument(
        "--active_buffer_size",
        type=int,
        default=32768,
        help="per-seq attended KV window in TOKENS (recall mode); the evict->recall "
        "cycle works to active_buffer_size - max_chunk_size",
    )
    p.add_argument(
        "--n_working",
        type=int,
        default=8192,
        help="protected recent window in TOKENS, never evicted, and the BM25 "
        "prefilter's term window",
    )
    p.add_argument(
        "--evict_tau",
        type=float,
        default=None,
        help="tau for the EVICTION bar; unset pins it to --recall_tau so one bar "
        "governs both directions",
    )
    p.add_argument(
        "--mass_reference_length",
        type=int,
        default=0,
        help="TOKENS the eviction/recall density is stated against: a key holding a "
        "uniform share of a context this long scores exactly 1, the same parity the "
        "taus are read against, and a buffer longer than this scores below it. 0 "
        "states the density against each query's OWN attended key count, under which "
        "uniform attention scores 1 at every length and growth applies no eviction "
        "pressure. Changes what the bar means, so the run header records it",
    )
    p.add_argument(
        "--recall_query_window",
        type=int,
        default=1,
        help="rerank query width in TOKENS; 1 = the most recent token alone",
    )
    p.add_argument("--recall_segment_size", type=int, default=256)
    p.add_argument(
        "--recall_bm25_query_tokens",
        type=int,
        default=0,
        help="BM25 prefilter query length in TOKENS, taken off the end of the stream; "
        "0 => --n_working, the whole protected window",
    )
    p.add_argument(
        "--block_radius",
        type=int,
        default=256,
        help="neighbourhood radius in TOKENS, a whole number of pages, 0 = off",
    )
    p.add_argument(
        "--active_position_layout",
        choices=["contiguous", "compacted"],
        default="contiguous",
        help="how active slot indices map to RoPE positions",
    )
    p.add_argument(
        "--short_offset",
        type=int,
        default=0,
        help="absolute RoPE position in TOKENS that demoted keys are roped to, and that "
        "the compacted layout collapses its distant region onto; 0 means the system "
        "prompt's length",
    )
    p.add_argument("--max_chunk_size", type=int, default=512)
    p.add_argument(
        "--decode_cycle_interval",
        type=int,
        default=0,
        help="generated tokens between the decode-cadence evict->recall cycles; 0 => "
        "--max_chunk_size, the prefill chunk budget",
    )
    p.add_argument(
        "--relevance_layers",
        type=str,
        default="",
        help="comma-separated top-most attention layers driving eviction density and "
        "the recall reranker; empty => default top quarter [3n/4, n)",
    )
    p.add_argument(
        "--recall_prefilter_k",
        type=int,
        default=24,
        help="cascade BM25 prefilter width in SEGMENTS; <=0 => no prefilter",
    )
    p.add_argument(
        "--recall_tau",
        type=float,
        default=None,
        help="cascade recall keep-bar tau in [0,1]. Required under --mode recall and "
        "--mode both, with no default, because 0 leaves nothing to recall and turns the "
        "run into an eviction-only measurement",
    )
    p.add_argument(
        "--attention_mass_decay",
        type=float,
        default=None,
        help="attention-mass EMA rate alpha, a memory length of 1/alpha TOKENS; unset "
        "derives 1/n_working so the horizon follows --n_working. Decay sets how long a "
        "recalled page survives, so a stated value is a mechanism change and the run "
        "header records which of the two it was",
    )
    p.add_argument(
        "--recall_temperature",
        type=float,
        default=0.0,
        help="softness of the recall keep-bar: 0 => a hard step",
    )
    p.add_argument(
        "--spill_dir",
        type=str,
        default="",
        help="spill-bucket dir; empty defaults to <cache_home>/pulsar/memory (recall mode only)",
    )
    p.add_argument(
        "--recall_buffer_size",
        type=int,
        default=0,
        help="per-seq candidate STAGING budget in TOKENS; 0 derives the worst-case "
        "scored set from recall_prefilter_k and recall_segment_size",
    )
    p.add_argument(
        "--ram_bucket_size",
        type=int,
        default=0,
        help="S1 (host-RAM) demoted-KV budget in TOKENS; 0 leaves S1 unbudgeted",
    )
    p.add_argument(
        "--recall_skip_fresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="filter pages evicted this cycle out of this cycle's candidate set",
    )
    p.add_argument(
        "--recall_during_prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run RECALL during PREFILL. Eviction is unaffected: the density bar "
        "always applies in both phases, since it is what holds residency at an "
        "equilibrium near the length the model attends over. A prefill cycle "
        "retrieves against the chunk already consumed rather than the one arriving, "
        "cannot select for a question that arrives last, and seeds what it promotes "
        "as PROTECTED",
    )
    p.add_argument("--dump_dir", type=Path, default=Path("scratchpad"))
    p.add_argument("--json", type=Path, default=None)
    return p


def parse_args() -> argparse.Namespace:
    """Flags with lengths_list and the engine derivations."""
    p = build_parser()
    args = p.parse_args()
    args.lengths_list = [x.strip() for x in args.lengths.split(",") if x.strip()]
    unknown = [x for x in args.lengths_list if x not in LENGTHS]
    if unknown:
        p.error(f"unknown length split(s) {unknown}; choose from {','.join(LENGTHS)}")
    if ceil_pages(args.active_buffer_size) > args.max_pages:
        p.error(
            f"active_buffer_size {args.active_buffer_size} needs "
            f"{ceil_pages(args.active_buffer_size)} pages, over --max_pages "
            f"{args.max_pages}"
        )
    derive_engine_args(args)
    return args


def print_banner(args: argparse.Namespace, modes: list[str], n_records: int) -> None:
    """Echo the run shape, the prompt and the engine configuration before any sample."""
    print(
        f"running task={args.task} (1 task of {len(TASKS)}) "
        f"lengths={args.lengths_list} ({len(args.lengths_list)} splits of "
        f"{len(LENGTHS)}) limit={args.limit} "
        f"modes={modes} on {n_records} records per split"
    )
    print(
        f"selection: max_pages={args.max_pages} "
        f"(={args.max_pages * PAGE_SIZE} tokens in FULL)"
    )
    print(
        f"config: gen_tokens={args.gen_tokens} think={args.think} "
        f"meta_prompt={args.meta_prompt} "
        f"native_window={args.native_window_tokens} "
        f"active_buffer_size={args.active_buffer_size} "
        f"n_working={args.n_working} "
        f"recall_query_window={args.recall_query_window} "
        f"recall_segment_size={args.recall_segment_size} "
        f"recall_bm25_query_tokens={args.recall_bm25_query_tokens or 'n_working'} "
        f"evict_tau={args.evict_tau} "
        f"mass_reference_length={args.mass_reference_length or 'attended'} "
        f"block_radius={args.block_radius} "
        f"active_position_layout={args.active_position_layout} "
        f"short_offset={args.short_offset or 'n_sink'} "
        f"max_chunk_size={args.max_chunk_size} "
        f"decode_cycle_interval={args.decode_cycle_interval}"
    )
    print(
        f"relevance_layers={args.relevance_layers_list or 'default[3n/4,n)'} "
        f"recall_prefilter_k={args.recall_prefilter_k} "
        f"recall_tau={args.recall_tau} "
        f"recall_temperature={args.recall_temperature} "
        f"attention_mass_decay={args.attention_mass_decay:.6e} "
        f"({args.attention_mass_decay_source})"
    )
    print(
        f"sampler: temperature={args.temperature} top_p={args.top_p} "
        f"top_k={args.top_k} seed={args.seed}"
        f"{' (RANDOM, run is unreproducible)' if args.temperature > 0 and args.seed < 0 else ''}"
    )
    if "recall" in modes:
        print(
            f"spill_dir={args.spill_dir} recall_buffer_size={staging_tokens(args)} "
            f"ram_bucket_size={args.ram_bucket_size} "
            f"recall_skip_fresh={args.recall_skip_fresh}"
        )
    print("system prompt:")
    print(build_system_text(args.task, META_PROMPTS[args.meta_prompt]))


def run_records(
    splits: dict[str, list[dict]],
    modes: list[str],
    loaded: PulsarModel,
    prompter: ChatMLConversation,
    args: argparse.Namespace,
) -> list[SampleResult]:
    """Every record of every split in every mode, one progress line per sample.

    Each result is appended to `--json`'s sibling .jsonl as it completes, flushed, so a
    long run can be read while it is still going. The final .json is still written at the
    end and stays the canonical artifact; the .jsonl is a live view of the same records.
    """
    results: list[SampleResult] = []
    stream = None
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        stream = args.json.with_suffix(".jsonl").open("w")
    total = sum(len(v) for v in splits.values()) * len(modes)
    done = 0
    t_start = time.perf_counter()
    for length in args.lengths_list:
        for index, rec in enumerate(splits[length]):
            for mode in modes:
                t0 = time.perf_counter()
                r = run_sample(
                    loaded, prompter, args.task, length, index, rec, mode, args
                )
                results.append(r)
                if stream is not None:
                    stream.write(json.dumps(asdict(r)) + "\n")
                    stream.flush()
                done += 1
                dt = time.perf_counter() - t0
                elapsed = time.perf_counter() - t_start
                eta = elapsed / done * (total - done)
                flag = r.skip_reason or ("OK" if r.correct else "x")
                print(
                    f"[{done}/{total}] {mode:6s} {r.id} len={length} "
                    f"toks={r.prompt_tokens} gen={r.gen_tokens} gold={r.gold} "
                    f"pred={r.pred} demoted={r.demoted} {flag} "
                    f"| {dt:.0f}s elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
                )
    if stream is not None:
        stream.close()
    return results


def report(
    results: list[SampleResult], modes: list[str], args: argparse.Namespace
) -> None:
    """Per-mode length curves, the distribution probes, the raw dump and the JSON."""
    for mode in modes:
        summarize(results, mode, args)
    report_curve(results, args)
    report_support_residency(results)
    report_histograms(results)
    report_density_by_position(results)
    dump_raw(results, args.dump_dir)
    report_skips(results, lambda r: f"{r.mode}/{r.length_split}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nwrote per-sample results -> {args.json}")


def main() -> None:
    args = parse_args()
    splits = {
        length: load_split(args.data_dir, args.task, length, args.limit)
        for length in args.lengths_list
    }
    modes = ["full", "recall"] if args.mode == "both" else [args.mode]

    loaded = checkpoint.load(checkpoint.model_path(args.model), device=args.device)
    print(
        f"loaded model={args.model} archive={checkpoint.model_path(args.model)} "
        f"eos_id={loaded.config.eos_id}"
    )
    args.native_window_tokens = native_window(args, loaded.config)

    print_banner(args, modes, min(len(v) for v in splits.values()))

    results = run_records(splits, modes, loaded, ChatMLConversation(), args)
    report(results, modes, args)


if __name__ == "__main__":
    main()
