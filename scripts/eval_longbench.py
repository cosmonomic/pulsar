"""LongBench v2 long-context multiple-choice QA harness for the C++ engine.

Drives the Engine surface (create / feed / step / stats / release) over LongBench
v2 records, under the LongBench v2 CoT protocol: a system-prompt sink asks the
model to reason step by step and end with "The correct answer is (X)"; the
extracted letter is graded by exact match. Recall mode needs a model that exposes
pre-RoPE Q (Qwen3).

Two memory regimes:

  FULL:   recall off; the active buffer is sized to the whole prompt so the entire
          context is attended. Prompts that would exceed --max_pages are skipped
          ("too_long_for_full").
  RECALL: recall on with a bounded active buffer (--active_buffer_size tokens).
          Eviction demotes over-window KV during chunked prefill; at generate-entry
          the engine recalls page-aligned segments into the freed headroom
          (recall_tau and recall_temperature decide how much of it is taken).
          The buffer is sized to the bounded window, not the full prompt, so a long
          prompt runs in a KV pool far smaller than FULL would need.

The system turn is the protected sink (n_sink tokens passed to feed), so eviction
never touches the instructions that make CoT work.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import Field, asdict, dataclass, field, replace
from pathlib import Path
from typing import ClassVar, Protocol

import torch
from xdg_base_dirs import xdg_cache_home

from pulsar import checkpoint
from pulsar.checkpoint import PulsarModel, Tokenizer
from pulsar.conversation import ChatMLConversation
from pulsar.rt import Pulsar, Session

# Line-buffer stdout so progress prints flush when output is redirected to a file
# (a non-TTY block-buffers by default).
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

DATA_URL = "https://huggingface.co/datasets/zai-org/LongBench-v2/resolve/main/data.json"
PAGE_SIZE = 16  # tensor-core paged attention supports page_size in {16, 32}

SYSTEM = (
    "You are a careful assistant answering a multiple-choice question about a "
    "long document. Think step by step, then end your reply with a line exactly "
    "like: The correct answer is (X)  -- where X is one of A, B, C, D."
)

# Metacognition clauses appended to SYSTEM, selected by --meta_prompt.
META_PROMPTS = {
    "none": "",
    "framework": (
        " Think out loud and in increments. Keep track of your reasoning and what it "
        "is for."
    ),
    "mental": (
        " Older detail you've read falls out of your active view -- but you can bring "
        "it back. When you mention the words, concepts, or the question a passage "
        "would answer, related content returns to your context. So when you need "
        "something you no longer see clearly, name what surrounds it, let it come "
        "back, and read what returns rather than trusting your memory of it."
    ),
    "mechanism": (
        " Your memory has two components:"
        " an active buffer containing most recent and relevant memories,"
        " and long-term bucket containing older memories."
        " To recall long-term memories,"
        " bring them back to the active buffer"
        " by thinking about related keywords and concepts."
        " Your memory system will find them using lexical similarity"
        " as a prefilter, and attention scoring."
        " To use your memory effectively,"
        " think in steps and out loud."
        " When necessary, spend more time thinking about related concepts"
        " which can help, but stop if nothing comes back."
        " Recalled memories arrive as fragments"
        " and may be mixed with other fragments,"
        " especially structured content may appear malformed."
    ),
}

# "answer is (X)" declaration. The (X) parens are required so prose like
# "the answer depends on ..." cannot match a stray a/b/c/d-initial word.
_ANSWER_DECL = re.compile(r"answer\s+is[\s:*]*\(([ABCD])\)", re.IGNORECASE)
_LETTER_PAREN = re.compile(r"\(([ABCD])\)")


def download_data(path: Path) -> None:
    """Fetch LongBench v2 data.json to path if absent. Exit on failure."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading LongBench v2 -> {path}")
    try:
        urllib.request.urlretrieve(DATA_URL, path)
    except Exception as exc:  # noqa: BLE001 - report and stop, do not fake data
        print(f"download failed: {exc}")
        print(f"fetch manually from {DATA_URL} to {path}")
        sys.exit(1)


def build_prompt_text(rec: dict) -> str:
    """LongBench v2 CoT user-message body: choices then a reason-then-answer tail."""
    return (
        f"{rec['context']}\n\n"
        f"What is the correct answer to this question: {rec['question']}\n"
        f"Choices:\n"
        f"(A) {rec['choice_A']}\n"
        f"(B) {rec['choice_B']}\n"
        f"(C) {rec['choice_C']}\n"
        f"(D) {rec['choice_D']}\n\n"
        f"Let's think step by step, then give the final answer as "
        f"'The correct answer is (X)'."
    )


def encode_prompt(
    tok: Tokenizer, prompter: ChatMLConversation, rec: dict, meta: str = ""
) -> tuple[list[int], list[int]]:
    """ChatML system + user turns plus the open assistant turn, as token ids.

    Returns (system_ids, body_ids): lengths for sizing the active/recall buffers
    and the truncation budget. The full stream is encoded once and split at the
    system length, so tokenization is identical to a single encode.
    """
    system_text = prompter.turn_open("system") + SYSTEM + meta + prompter.turn_close()
    n_sink = len(tok.encode(system_text).ids)
    text = (
        system_text
        + prompter.turn_open("user")
        + build_prompt_text(rec)
        + prompter.turn_close()
        + prompter.turn_open("assistant")
    )
    full = tok.encode(text).ids
    return full[:n_sink], full[n_sink:]


def extract_letter(text: str) -> str | None:
    """Final-answer letter, or None if the model committed to none.

    With a closed thinking block only the post-</think> region is read: an explicit
    "answer is (X)" declaration first, then the last standalone (X). Without a closed
    block only a declaration counts, never a stray mid-reasoning (X).
    """
    if not text:
        return None
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[-1]
        decls = _ANSWER_DECL.findall(tail)
        if decls:
            return decls[-1].upper()
        parens = _LETTER_PAREN.findall(tail)
        return parens[-1].upper() if parens else None
    decls = _ANSWER_DECL.findall(text)
    return decls[-1].upper() if decls else None


def derive_engine_args(args: argparse.Namespace) -> None:
    """Fill in the engine settings build_engine reads that no flag carries directly.

    Shared by every harness: --recall_tau carries no default, so a mode that recalls
    must state it. Only --mode full, which never recalls, gets a value filled in.
    --attention_mass_decay is derived from --n_working unless stated, and every
    derivation lands on args before the run header prints, so the header records the
    values that ran.
    """
    args.relevance_layers_list = [
        int(x) for x in args.relevance_layers.split(",") if x.strip()
    ]
    if args.n_working <= 0:
        raise SystemExit(f"--n_working must be > 0, got {args.n_working}")
    # The EMA memory length is 1/alpha TOKENS, so alpha has to follow n_working or a
    # run that varies the protected window silently varies the horizon with it.
    args.attention_mass_decay_source = "explicit"
    if args.attention_mass_decay is None:
        args.attention_mass_decay = 1.0 / args.n_working
        args.attention_mass_decay_source = "1/n_working"
    # 0 asks for the engine's own fallback, max_chunk_size. Resolved here so the run
    # header records the interval that ran.
    if args.decode_cycle_interval <= 0:
        args.decode_cycle_interval = args.max_chunk_size
    if not args.spill_dir:
        args.spill_dir = str(xdg_cache_home() / "pulsar" / "memory")
    if args.recall_tau is None:
        if args.mode != "full":
            raise SystemExit(
                f"--recall_tau is required under --mode {args.mode}: it has no default "
                "because tau 0 disables recall entirely -- the keep-bar demotes "
                "nothing, so all eviction falls to the capacity backstop, which lands "
                "the active buffer exactly on its ceiling and leaves no capacity to "
                "recall into. Such a run reports mode=recall with recalled_pages=0 and "
                "measures eviction only. Pass --recall_tau 0 to ask for that "
                "deliberately."
            )
        args.recall_tau = 0.0
    # One bar governs both directions unless a run deliberately splits them. The engine
    # keeps the two taus independent, so the unification lives here and the run header
    # records the number that ran.
    if args.evict_tau is None:
        args.evict_tau = args.recall_tau


def build_engine(
    loaded: PulsarModel,
    args: argparse.Namespace,
    active_buffer_size: int,
    recall_buffer_size: int,
) -> Pulsar:
    """A one-sequence engine over loaded's weights, configured from args.

    The two buffer sizes are in TOKENS and are derived per sample, so they are passed
    rather than read off args. derive_engine_args must have run.
    """
    cfg = loaded.config
    return Pulsar(
        loaded.weights,
        {
            "model": args.model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "head_dim": cfg.head_dim,
            "hidden": cfg.hidden_size,
            "intermediate": cfg.intermediate_size,
            "vocab": cfg.vocab_size,
            "rope_theta": float(cfg.rope_theta),
            "rms_eps": float(cfg.rms_norm_eps),
            "active_buffer_size": active_buffer_size,
            "page_size": PAGE_SIZE,
            "device": args.device,
            "max_sessions": 1,
            "max_running": 1,
            "max_chunk_size": args.max_chunk_size,
            "decode_cycle_interval": args.decode_cycle_interval,
            "eos_id": cfg.eos_id,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "n_working": args.n_working,
            "evict_tau": args.evict_tau,
            "mass_reference_length": args.mass_reference_length,
            "recall_query_window": args.recall_query_window,
            "recall_segment_size": args.recall_segment_size,
            "block_radius": args.block_radius,
            "active_position_layout": args.active_position_layout,
            "short_offset": args.short_offset,
            "recall_prefilter_k": args.recall_prefilter_k,
            "recall_bm25_query_tokens": args.recall_bm25_query_tokens,
            "recall_tau": args.recall_tau,
            "recall_temperature": args.recall_temperature,
            "attention_mass_decay": args.attention_mass_decay,
            "recall_buffer_size": recall_buffer_size,
            "ram_bucket_size": args.ram_bucket_size,
            "recall_skip_fresh": args.recall_skip_fresh,
            "recall_during_prefill": args.recall_during_prefill,
            "spill_dir": args.spill_dir,
        }
        # An empty list has no element type for TorchBind to infer; omit it and let
        # the engine apply its default. A non-empty list carries its int type.
        | (
            {"relevance_layers": args.relevance_layers_list}
            if args.relevance_layers_list
            else {}
        ),
        loaded.tokenizer,
        dtype=torch.bfloat16,
    )


class ProbedSample(Protocol):
    """Structural view of a harness sample result: the engine probes and the row filter.

    Each harness declares its own SampleResult with its own identity fields, so the
    helpers shared between harnesses are typed against this instead of one of them.
    """

    # dataclasses.replace() takes a dataclass, so a type variable bound to this
    # protocol has to carry the dataclass marker.
    __dataclass_fields__: ClassVar[dict[str, Field[object]]]

    mode: str
    skip_reason: str | None
    density_hist: list[int]
    density_log10_lo: int
    density_log10_hi: int
    recall_hist: list[int]
    recall_hist_decode: list[int]
    decode_cycles: int
    post_evict_tokens: int
    post_evict_tokens_decode: int
    post_recall_tokens: int
    post_recall_tokens_decode: int
    starved_cycles: int
    starved_cycles_decode: int
    recall_lo: int
    recall_hi: int
    position_raw_mass_sum: list[float]
    position_corrected_density_sum: list[float]
    position_page_count: list[int]


@dataclass
class SampleResult:
    id: str
    length: str
    domain: str
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
    # Eviction-candidate density histogram: log10 bins over
    # [10**density_log10_lo, 10**density_log10_hi], bin 0 absorbs the underflow and
    # the last bin the overflow. The binned value is the length-normalized density
    # the eviction bar is compared against, in units where uniform attention is
    # 1/active_buffer_size.
    density_hist: list[int] = field(default_factory=list)
    density_log10_lo: int = 0
    density_log10_hi: int = 0
    # Page-age histograms in TOKENS (stream_len minus the page's base address): log10
    # bins over [10**age_log10_lo, 10**age_log10_hi], bin 0 absorbing the underflow and
    # the last bin the overflow. evicted_age_hist covers the demoted pages,
    # candidate_age_hist every evictable page, so the two compare.
    evicted_age_hist: list[int] = field(default_factory=list)
    evicted_age_hist_decode: list[int] = field(default_factory=list)
    candidate_age_hist: list[int] = field(default_factory=list)
    candidate_age_hist_decode: list[int] = field(default_factory=list)
    age_log10_lo: int = 0
    age_log10_hi: int = 0
    # Rerank-score histogram: linear bins over [recall_lo, recall_hi] of every finite
    # score as scored (before selection), so it is independent of
    # recall_tau. Scores are squashed to s = r/(1+r), which puts the keep-bar at s = tau.
    # The _decode variant counts the generation-phase cycles alone; those are a few
    # percent of the scored mass, so the pooled histogram is a prefill measurement.
    recall_hist: list[int] = field(default_factory=list)
    recall_hist_decode: list[int] = field(default_factory=list)
    recall_lo: int = 0
    recall_hi: int = 0
    # Density-by-position probe: per-bucket SUMS over all pages of every eviction,
    # bucketed by normalized position, before and after the age bias correction.
    # position_page_count carries the pages behind each bucket; means come from
    # dividing the sums by it, so aggregating across samples sums both.
    position_raw_mass_sum: list[float] = field(default_factory=list)
    position_corrected_density_sum: list[float] = field(default_factory=list)
    position_page_count: list[int] = field(default_factory=list)


def ceil_pages(n_tokens: int) -> int:
    """Pages a token count occupies, rounding a partial page up."""
    return -(-n_tokens // PAGE_SIZE)


def staging_tokens(args: argparse.Namespace) -> int:
    """Per-seq RecallBuffer size in TOKENS: the worst-case simultaneously scored set.

    A cycle scores the prefilter's top-k segments plus a rerank halo per kept segment,
    all resident at once, and the engine hard-errors rather than clamping, so this is
    the buffer's floor. An explicit --recall_buffer_size overrides it. Without a
    prefilter there is no bound to derive.
    """
    if args.recall_buffer_size > 0:
        return args.recall_buffer_size
    if args.recall_prefilter_k <= 0:
        raise SystemExit(
            "--recall_buffer_size is required when --recall_prefilter_k <= 0: "
            "without a prefilter the scored set is the whole conversation"
        )
    # Only kept-segment pages are staged, so the worst-case scored set is exactly the
    # prefilter's winners. Expansion pulls neighbours straight out of the bucket and
    # never stages them.
    segment_size = args.recall_segment_size or PAGE_SIZE
    return args.recall_prefilter_k * segment_size


def _plan_buffers(
    mode: str, n_prompt: int, args: argparse.Namespace
) -> tuple[int, int] | str:
    """(active_buffer_size, recall_buffer_size) in TOKENS, or a skip reason.

    A 0 band bound is no bound on that side. FULL keeps the whole prompt plus
    generation attended; RECALL sizes the buffer to the bounded window and the
    evict->recall cycle works to active_buffer_size - max_chunk_size.
    """
    if args.min_prompt_tokens and n_prompt < args.min_prompt_tokens:
        return "below_band"
    if args.max_prompt_tokens and n_prompt >= args.max_prompt_tokens:
        return "above_band"
    if mode == "full":
        recall_buffer_size = 0
        need_pages = ceil_pages(n_prompt + args.gen_tokens) + 2
        if need_pages > args.max_pages:
            return "too_long_for_full"
    else:
        recall_buffer_size = staging_tokens(args)
        need_pages = ceil_pages(args.active_buffer_size)
    return need_pages * PAGE_SIZE, recall_buffer_size


async def _active_pages(session: Session) -> int:
    try:
        return len((await session.get_context()).keys)
    except Exception:
        return 0


async def _dump_context(session: Session, rid: str, prompt_tokens: list[int]) -> None:
    """Dump the resident and demoted page ids and tokens when PULSAR_DUMP_CTX == rid."""
    if os.environ.get("PULSAR_DUMP_CTX") != rid:
        return
    ctx = await session.get_context()
    hist = await session.get_history()
    with open(f"scratchpad/ctxdump_{rid[:8]}.json", "w") as fh:
        json.dump(
            {
                "ctx_page_ids": list(ctx.keys),
                "ctx_tokens": list(ctx.tokens()),
                "hist_page_ids": list(hist.keys),
                "hist_tokens": list(hist.tokens()),
                "prompt_tokens": prompt_tokens,
            },
            fh,
        )


def _with_stats[SampleResultT: ProbedSample](
    result: SampleResultT, st: dict
) -> SampleResultT:
    """result carrying the recall counters and histograms from a session.get_stats() dict.

    get_stats() is heterogeneous: scalars are int64, the histograms int64 or float64 lists.
    """
    return replace(
        result,
        evicted_pages=int(st["evicted_pages"]),
        bar_evicted_pages=int(st["bar_evicted_pages"]),
        bar_evicted_pages_decode=int(st["bar_evicted_pages_decode"]),
        backstop_evicted_pages=int(st["backstop_evicted_pages"]),
        backstop_evicted_pages_decode=int(st["backstop_evicted_pages_decode"]),
        raw_mass_victim_pages=int(st["raw_mass_victim_pages"]),
        raw_mass_victim_pages_decode=int(st["raw_mass_victim_pages_decode"]),
        raw_mass_agree_pages=int(st["raw_mass_agree_pages"]),
        raw_mass_agree_pages_decode=int(st["raw_mass_agree_pages_decode"]),
        recalled_pages=int(st["recalled_pages"]),
        cycles=int(st["cycles"]),
        decode_cycles=int(st["decode_cycles"]),
        post_evict_tokens=int(st["post_evict_tokens"]),
        post_evict_tokens_decode=int(st["post_evict_tokens_decode"]),
        post_recall_tokens=int(st["post_recall_tokens"]),
        post_recall_tokens_decode=int(st["post_recall_tokens_decode"]),
        starved_cycles=int(st["starved_cycles"]),
        starved_cycles_decode=int(st["starved_cycles_decode"]),
        staged_hits=int(st["staged_hits"]),
        staged_misses=int(st["staged_misses"]),
        promoted_s1=int(st["promoted_s1"]),
        promoted_s2=int(st["promoted_s2"]),
        density_hist=[int(c) for c in st["density_hist"]],
        density_log10_lo=int(st["density_log10_lo"]),
        density_log10_hi=int(st["density_log10_hi"]),
        evicted_age_hist=[int(c) for c in st["evicted_age_hist"]],
        evicted_age_hist_decode=[int(c) for c in st["evicted_age_hist_decode"]],
        candidate_age_hist=[int(c) for c in st["candidate_age_hist"]],
        candidate_age_hist_decode=[int(c) for c in st["candidate_age_hist_decode"]],
        age_log10_lo=int(st["age_log10_lo"]),
        age_log10_hi=int(st["age_log10_hi"]),
        recall_hist=[int(c) for c in st["recall_hist"]],
        recall_hist_decode=[int(c) for c in st["recall_hist_decode"]],
        recall_lo=int(st["recall_lo"]),
        recall_hi=int(st["recall_hi"]),
        position_raw_mass_sum=[float(v) for v in st["position_raw_mass_sum"]],
        position_corrected_density_sum=[
            float(v) for v in st["position_corrected_density_sum"]
        ],
        position_page_count=[int(c) for c in st["position_page_count"]],
    )


def skip_counts(rows: Sequence[ProbedSample]) -> dict[str, int]:
    """Skips by reason over rows, ordered by reason."""
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.skip_reason is not None:
            counts[r.skip_reason] += 1
    return dict(sorted(counts.items()))


def format_skips(counts: Mapping[str, int]) -> str:
    """reason=count pairs, or "none"."""
    return ",".join(f"{k}={v}" for k, v in counts.items()) if counts else "none"


def report_skips[SampleResultT: ProbedSample](
    results: Sequence[SampleResultT], key: Callable[[SampleResultT], str]
) -> None:
    """Every skipped sample, grouped by key, printed last.

    A skip drops a record out of an average without moving the average, and the
    records a page ceiling drops are the longest ones, so an unread skip count biases
    the result toward the short end. Any group with a skip is named here.
    """
    total = sum(r.skip_reason is not None for r in results)
    print("\n=== skipped samples ===")
    if not total:
        print(f"  none: all {len(results)} sample(s) scored")
        return
    print(f"  WARNING: {total} of {len(results)} sample(s) were NOT scored")
    groups: dict[str, list[SampleResultT]] = defaultdict(list)
    for r in results:
        groups[key(r)].append(r)
    width = max(len(name) for name in groups)
    for name in sorted(groups):
        counts = skip_counts(groups[name])
        if not counts:
            continue
        n_skipped = sum(counts.values())
        print(
            f"  {name:>{width}}  scored={len(groups[name]) - n_skipped:<4d} "
            f"skipped={n_skipped:<4d} {format_skips(counts)}"
        )


def truncate_body(body_ids: list[int], budget: int) -> list[int]:
    """Drop the middle of the user turn so it fits budget tokens.

    The question, the choices and the open assistant turn are the LAST tokens of the body,
    so the tail is what must survive; the head carries the document's opening. Removing the
    middle is the standard LongBench truncation and it mirrors what eviction protects: a
    sink at the front and a working window at the end. budget <= 0 disables truncation.
    """
    if budget <= 0 or len(body_ids) <= budget:
        return body_ids
    head = budget // 2
    return body_ids[:head] + body_ids[len(body_ids) - (budget - head) :]


def run_sample(
    loaded: PulsarModel,
    prompter: ChatMLConversation,
    rec: dict,
    mode: str,
    args: argparse.Namespace,
) -> SampleResult:
    system_ids, body_ids = encode_prompt(
        loaded.tokenizer, prompter, rec, META_PROMPTS[args.meta_prompt]
    )
    body_ids = truncate_body(body_ids, args.truncate_tokens - len(system_ids))
    n_prompt = len(system_ids) + len(body_ids)
    gold = str(rec["answer"]).strip().upper()
    rid = str(rec["_id"])
    length = str(rec["length"])
    domain = str(rec.get("domain", ""))

    def skip(reason: str) -> SampleResult:
        return SampleResult(
            id=rid,
            length=length,
            domain=domain,
            mode=mode,
            gold=gold,
            prompt_tokens=n_prompt,
            pred=None,
            correct=False,
            gen_tokens=0,
            demoted=0,
            skip_reason=reason,
        )

    plan = _plan_buffers(mode, n_prompt, args)
    if isinstance(plan, str):
        return skip(plan)
    active_buffer_size, recall_buffer_size = plan

    async def _run() -> tuple[list[int], int, dict, int]:
        pulsar = build_engine(loaded, args, active_buffer_size, recall_buffer_size)
        try:
            session = await pulsar.create_session(system_ids)
            response = await session.send(body_ids, args.gen_tokens)
            out = [tok async for tok in response]
            demoted = await session.get_demoted_tokens_count()
            st = await session.get_stats()
            active_pages = await _active_pages(session)
            await _dump_context(session, rid, system_ids + body_ids)
            await session.destroy()
            return out, demoted, st, active_pages
        finally:
            await pulsar.aclose()

    try:
        out, demoted, st, active_pages = asyncio.run(_run())
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return skip("OOM")

    decoded = loaded.tokenizer.decode(out)
    pred = extract_letter(decoded)
    return _with_stats(
        SampleResult(
            id=rid,
            length=length,
            domain=domain,
            mode=mode,
            gold=gold,
            prompt_tokens=n_prompt,
            pred=pred,
            correct=pred == gold,
            gen_tokens=len(out),
            demoted=demoted,
            active_pages=active_pages,
            skip_reason=None,
            raw=decoded,
        ),
        st,
    )


def summarize(results: list[SampleResult], mode: str, gen_cap: int) -> None:
    rows = [r for r in results if r.mode == mode]
    scored = [r for r in rows if r.skip_reason is None]
    skips = skip_counts(rows)
    n_correct = sum(r.correct for r in scored)
    acc = n_correct / len(scored) if scored else 0.0
    total_demoted = sum(r.demoted for r in scored)
    # no_answer: no letter declared. Counted as wrong and reported separately.
    n_no_answer = sum(r.pred is None for r in scored)
    n_hit_cap = sum(r.gen_tokens >= gen_cap for r in scored)

    print(f"\n=== mode={mode} ===")
    print(f"n_scored={len(scored)}  correct={n_correct}  accuracy={acc:.3f}")
    print(f"n_no_answer={n_no_answer}  n_hit_gen_cap={n_hit_cap}")
    print(f"n_skipped={sum(skips.values())} of {len(rows)}: {format_skips(skips)}")
    if mode == "recall":
        print(f"total_demoted_kv={total_demoted}")
        cand = sum(r.staged_hits + r.staged_misses for r in scored)
        promoted = sum(r.promoted_s1 + r.promoted_s2 for r in scored)
        hit_rate = sum(r.staged_hits for r in scored) / cand if cand else 0.0
        s2_share = sum(r.promoted_s2 for r in scored) / promoted if promoted else 0.0
        print(f"staging_hit_rate={hit_rate:.3f} over {cand} candidates")
        print(f"s2_share_of_promotions={s2_share:.3f} over {promoted} promotions")

    # The length bucket is the axis --max_pages skews, so it carries the skip counts.
    by_len: dict[str, list[SampleResult]] = defaultdict(list)
    for r in rows:
        by_len[r.length].append(r)
    by_dom: dict[str, list[SampleResult]] = defaultdict(list)
    for r in scored:
        by_dom[r.domain].append(r)
    print("by length:")
    for k in sorted(by_len):
        g = [x for x in by_len[k] if x.skip_reason is None]
        bucket_skips = skip_counts(by_len[k])
        acc = sum(x.correct for x in g) / len(g) if g else float("nan")
        print(
            f"  {k:8s} n={len(g):3d} acc={acc:.3f} "
            f"skipped={sum(bucket_skips.values()):3d} {format_skips(bucket_skips)}"
        )
    if scored:
        print("by domain:")
        for k in sorted(by_dom):
            g = by_dom[k]
            print(
                f"  {k:28s} n={len(g):3d} acc={sum(x.correct for x in g) / len(g):.3f}"
            )


def _sum_hists(hists: list[list[int]]) -> list[int]:
    """Elementwise sum of equal-length histograms ([] when there are none)."""
    if not hists:
        return []
    n = len(hists[0])
    if any(len(h) != n for h in hists):
        raise SystemExit("histogram bin counts differ across samples")
    return [sum(h[b] for h in hists) for b in range(n)]


def hist_median(hist: list[int], log10_lo: int, log10_hi: int) -> float:
    """Median of a log10-binned histogram, read at the containing bin's centre.

    Resolution is one bin, so the value only ever names the bin the median falls in.
    NaN when the histogram is empty.
    """
    total = sum(hist)
    if not total:
        return float("nan")
    width = (log10_hi - log10_lo) / len(hist)
    seen = 0
    for b, count in enumerate(hist):
        seen += count
        if 2 * seen >= total:
            return 10.0 ** (log10_lo + width * (b + 0.5))
    return float("nan")


def _print_hist(title: str, edges: list[float], counts: list[int], fmt: str) -> None:
    """One histogram with both cumulative directions.

    cum<hi is the fraction strictly below the bin's upper edge, >=lo the fraction at
    or above its lower edge.
    """
    total = sum(counts)
    print(f"\n=== {title} ===")
    if total == 0:
        print("  no observations")
        return
    print(f"  n={total}")
    print(
        f"  {'bin':>3} {'lo':>10} {'hi':>10} {'count':>9} "
        f"{'frac':>7} {'cum<hi':>7} {'>=lo':>7}"
    )
    cum = 0
    for b, c in enumerate(counts):
        below = cum
        cum += c
        print(
            f"  {b:>3} {edges[b]:>10{fmt}} {edges[b + 1]:>10{fmt}} {c:>9} "
            f"{c / total:>7.3f} {cum / total:>7.3f} {1.0 - below / total:>7.3f}"
        )


def report_histograms(results: Sequence[ProbedSample]) -> None:
    """Density and rerank-score distributions pooled over the scored recall samples."""
    rows = [
        r
        for r in results
        if r.mode == "recall" and r.skip_reason is None and r.density_hist
    ]
    if not rows:
        return
    dens = _sum_hists([r.density_hist for r in rows])
    lo, hi = rows[0].density_log10_lo, rows[0].density_log10_hi
    n = len(dens)
    _print_hist(
        f"eviction-candidate density ({n} log10 bins over "
        f"[1e{lo}, 1e{hi}], the derived eviction bar is compared against this)",
        [10.0 ** (lo + (hi - lo) * b / n) for b in range(n + 1)],
        dens,
        ".2e",
    )
    rer = _sum_hists([r.recall_hist for r in rows])
    rlo, rhi = rows[0].recall_lo, rows[0].recall_hi
    n = len(rer)
    _print_hist(
        f"rerank score as scored ({n} linear bins over [{rlo}, {rhi}], before "
        "selection)",
        [rlo + (rhi - rlo) * b / n for b in range(n + 1)],
        rer,
        ".2f",
    )
    # Decode alone. Generation cycles are a few percent of the scored mass, so the
    # pooled histogram above answers a prefill question however the decode cycles went.
    rer_d = _sum_hists([r.recall_hist_decode for r in rows])
    if sum(rer_d):
        _print_hist(
            f"rerank score as scored, DECODE cycles only ({n} linear bins over "
            f"[{rlo}, {rhi}], before selection)",
            [rlo + (rhi - rlo) * b / n for b in range(n + 1)],
            rer_d,
            ".2f",
        )


def report_density_by_position(results: Sequence[ProbedSample]) -> None:
    """Per-page density by normalized position, before and after the bias correction.

    Every page of every eviction is counted, the sink prefix and the working window
    included. Means come from summed sums over summed page counts.
    """
    rows = [
        r
        for r in results
        if r.mode == "recall" and r.skip_reason is None and r.position_page_count
    ]
    if not rows:
        return
    n = len(rows[0].position_page_count)
    if any(len(r.position_page_count) != n for r in rows):
        raise SystemExit("position bucket counts differ across samples")
    raw = [sum(r.position_raw_mass_sum[b] for r in rows) for b in range(n)]
    corrected = [
        sum(r.position_corrected_density_sum[b] for r in rows) for b in range(n)
    ]
    pages = [sum(r.position_page_count[b] for r in rows) for b in range(n)]
    total = sum(pages)
    print(f"\n=== per-page density by normalized position ({n} buckets) ===")
    if total == 0:
        print("  no observations")
        return
    print(f"  pages={total}")
    print(
        f"  {'bucket':>6} {'pos_lo':>7} {'pos_hi':>7} {'mean_raw_mass':>14} "
        f"{'mean_corrected':>15} {'ratio':>7} {'pages':>9}"
    )
    for b in range(n):
        mean_raw = raw[b] / pages[b] if pages[b] else 0.0
        mean_corr = corrected[b] / pages[b] if pages[b] else 0.0
        ratio = mean_corr / mean_raw if mean_raw > 0.0 else 0.0
        print(
            f"  {b:>6} {b / n:>7.3f} {(b + 1) / n:>7.3f} {mean_raw:>14.4e} "
            f"{mean_corr:>15.4e} {ratio:>7.2f} {pages[b]:>9}"
        )


def report_recall_advantage(results: list[SampleResult]) -> None:
    """Samples RECALL scored that FULL could not (too_long/OOM)."""
    full = {r.id: r for r in results if r.mode == "full"}
    recall = {r.id: r for r in results if r.mode == "recall"}
    wins = [
        rid
        for rid in recall
        if recall[rid].skip_reason is None
        and full.get(rid)
        and full[rid].skip_reason is not None
    ]
    print("\n=== RECALL vs FULL feasibility ===")
    if wins:
        print(f"RECALL scored {len(wins)} sample(s) FULL skipped:")
        for rid in wins:
            print(f"  {rid} (full skip={full[rid].skip_reason})")
    else:
        print("no samples RECALL scored that FULL could not")


def dump_raw(results: list[SampleResult], dump_dir: Path, per_mode: int = 2) -> None:
    """Write a few raw CoT outputs per mode to dump_dir for inspection."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    out = dump_dir / "raw_cot_samples.txt"
    lines: list[str] = []
    for mode in ("full", "recall"):
        picked = [r for r in results if r.mode == mode and r.raw is not None][:per_mode]
        for r in picked:
            lines.append(
                f"===== mode={mode} id={r.id} gold={r.gold} pred={r.pred} ====="
            )
            lines.append(r.raw or "")
            lines.append("")
    if lines:
        out.write_text("\n".join(lines))
        print(f"\nwrote raw CoT samples -> {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="qwen3-8b")
    p.add_argument("--data", type=Path, default=Path("data/longbench_v2.json"))
    p.add_argument(
        # Required, not defaulted: this filters the record set, so a default would
        # silently scope a run. An --id list spanning buckets keeps only its members
        # of the chosen bucket.
        "--length", choices=["short", "medium", "long", "all"], required=True
    )
    p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="records taken from the head of the filtered set. IGNORED when --id is "
        "given, which selects by id instead; 0 selects nothing and the run exits",
    )
    p.add_argument(
        "--id",
        type=str,
        default=None,
        help="run only ids with this prefix (comma-separated)",
    )
    p.add_argument(
        "--meta_prompt",
        choices=list(META_PROMPTS),
        default="none",
        help="metacognition clause appended to the system prompt",
    )
    p.add_argument("--domain", type=str, default=None, help="case-insensitive substr")
    p.add_argument("--difficulty", choices=["easy", "hard", "all"], default="all")
    p.add_argument("--mode", choices=["full", "recall", "both"], default="full")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--gen_tokens", type=int, default=512)
    # Decode sampler. Distinct from --recall_temperature, which softens the recall
    # keep-bar and does not touch decoding.
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="decode sampling temperature; 0 => greedy (argmax). Greedy decoding is what "
        "makes a repeating chain of thought a fixed point, since the recall query is the "
        "recent generated text",
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
        help="KV pool ceiling in pages: a FULL sample whose prompt plus generation "
        "needs more is skipped as too_long_for_full (that skip is the RECALL vs FULL "
        "feasibility result). A recall-mode buffer over it is a startup error. This "
        "ceiling removes the LONGEST records, so read the skip report before the "
        "accuracy",
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
        "prefilter's term window. The engine has no default for it; 8192 is this "
        "harness's pinned operating point",
    )
    p.add_argument(
        "--evict_tau",
        type=float,
        default=None,
        help="tau for the EVICTION bar; unset pins it to --recall_tau so one "
        "bar governs both directions. Resolved before the run header prints, so the "
        "logged value is always the one that ran",
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
    p.add_argument("--min_prompt_tokens", type=int, default=0)  # band filter lo (0=off)
    p.add_argument("--max_prompt_tokens", type=int, default=0)  # band filter hi (0=off)
    p.add_argument(
        "--truncate_tokens",
        type=int,
        default=0,
        help="cap the prompt at this many TOKENS by dropping the middle of the user turn, "
        "keeping the document's head and the question/choices tail; 0 = no truncation. "
        "This DISCARDS context rather than demoting it, so it is the baseline the recall "
        "mechanism has to beat: run it with --mode full and no eviction",
    )
    p.add_argument(
        "--recall_query_window",
        type=int,
        default=1,
        help="rerank query width in TOKENS; 1 = the most recent token alone. Independent "
        "of n_working, and it does not narrow the bm25 prefilter",
    )
    p.add_argument("--recall_segment_size", type=int, default=256)
    p.add_argument(
        "--block_radius",
        type=int,
        default=256,
        help="neighbourhood radius in TOKENS, a whole number of pages, 0 = off. The "
        "same radius governs both directions: eviction keeps a page whose neighbourhood "
        "clears the bar, and recall promotes a winning page's whole neighbourhood",
    )
    p.add_argument(
        "--active_position_layout",
        choices=["contiguous", "compacted"],
        default="contiguous",
        help="how active slot indices map to RoPE positions: contiguous (the slot index "
        "IS the position) or compacted (sink, a distant region collapsed onto "
        "short_offset, then the working window) -- the InfLLM/LM-Infinite layout, where "
        "distant context carries no relative order",
    )
    p.add_argument(
        "--short_offset",
        type=int,
        default=0,
        help="absolute RoPE position in TOKENS that demoted keys are roped to, and that "
        "the compacted layout collapses its distant region onto; 0 means the system "
        "prompt's length. A non-zero value must be >= that length. Read under both "
        "layouts",
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
        help="cascade BM25 prefilter width in SEGMENTS; <=0 => no prefilter (score "
        "every segment)",
    )
    p.add_argument(
        "--recall_bm25_query_tokens",
        type=int,
        default=0,
        help="BM25 prefilter query length in TOKENS, taken off the end of the stream; "
        "0 => --n_working, the whole protected window. Changes what is retrieved, so "
        "the run header records it",
    )
    p.add_argument(
        "--recall_tau",
        type=float,
        default=None,
        help="cascade recall keep-bar tau in [0,1]: the sigmoid of the page best-key "
        "attention odds against the working set (0.5 = parity). Required under --mode "
        "recall and --mode both, with no default, because 0 leaves nothing to recall "
        "and turns the run into an eviction-only measurement",
    )
    p.add_argument(
        "--attention_mass_decay",
        type=float,
        default=None,
        help="attention-mass EMA rate alpha: a page's mass converges to its per-forward "
        "density with a memory length of 1/alpha TOKENS. Unset derives 1/n_working, so "
        "the EMA remembers exactly the protected window and the horizon follows "
        "--n_working instead of drifting away from it. Decay sets how long a recalled "
        "page survives, so a stated value is a mechanism change and the run header "
        "records which of the two it was. 0 would derive "
        "1/(active_buffer_size - max_chunk_size) in the engine instead, which is also "
        "the floor; capped at 1",
    )
    p.add_argument(
        "--recall_temperature",
        type=float,
        default=0.0,
        help="softness of the recall keep-bar: 0 => a hard step (recall iff score >= "
        "recall_tau); >0 => an independent Bernoulli draw per page with "
        "probability sigmoid((logit(score) - logit(recall_tau)) / "
        "recall_temperature), best first until the cycle's capacity is full",
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
        help="per-seq candidate STAGING budget in TOKENS (relevance-layer K on the "
        "GPU); 0 derives the worst-case scored set from recall_prefilter_k and "
        "recall_segment_size",
    )
    p.add_argument(
        "--ram_bucket_size",
        type=int,
        default=0,
        help="S1 (host-RAM) demoted-KV budget in TOKENS, above which the coldest "
        "pages spill to spill_dir; 0 leaves S1 unbudgeted and never spills",
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
    """Flags with relevance_layers_list and spill_dir derived onto the namespace."""
    p = build_parser()
    args = p.parse_args()
    if ceil_pages(args.active_buffer_size) > args.max_pages:
        p.error(
            f"active_buffer_size {args.active_buffer_size} needs "
            f"{ceil_pages(args.active_buffer_size)} pages, over --max_pages "
            f"{args.max_pages}"
        )
    derive_engine_args(args)
    return args


def select_records(args: argparse.Namespace) -> list[dict]:
    """Records passing the length, domain, difficulty and id filters. Exit if none do.

    --limit applies only when --id is absent.
    """
    download_data(args.data)
    records = json.loads(args.data.read_text())
    if args.length != "all":
        records = [r for r in records if r["length"] == args.length]
    if args.domain:
        sub = args.domain.lower()
        records = [r for r in records if sub in r.get("domain", "").lower()]
    if args.difficulty != "all":
        records = [r for r in records if r.get("difficulty") == args.difficulty]
    if args.id:
        prefixes = tuple(p for p in args.id.split(",") if p)
        records = [
            r
            for r in records
            if str(r.get("_id", r.get("id", ""))).startswith(prefixes)
        ]
    else:
        records = records[: args.limit]
    if not records:
        print("no records matched the filters")
        sys.exit(1)
    return records


def print_banner(args: argparse.Namespace, modes: list[str], n_records: int) -> None:
    """Echo the filters and the engine configuration before any sample runs."""
    print(
        f"running length={args.length} limit={args.limit} id={args.id} "
        f"domain={args.domain} difficulty={args.difficulty} "
        f"modes={modes} on {n_records} records"
    )
    print(
        f"selection: min_prompt_tokens={args.min_prompt_tokens} "
        f"max_prompt_tokens={args.max_prompt_tokens} "
        f"truncate_tokens={args.truncate_tokens} "
        f"max_pages={args.max_pages} (={args.max_pages * PAGE_SIZE} tokens in FULL)"
    )
    print(
        f"config: gen_tokens={args.gen_tokens} "
        f"meta_prompt={args.meta_prompt} "
        f"active_buffer_size={args.active_buffer_size} "
        f"n_working={args.n_working} "
        f"recall_query_window={args.recall_query_window} "
        f"recall_segment_size={args.recall_segment_size} "
        f"recall_bm25_query_tokens={args.recall_bm25_query_tokens or 'n_working'} "
        f"evict_tau={args.evict_tau} "
        f"mass_reference_length={args.mass_reference_length or 'attended'} "
        f"block_radius={args.block_radius} "
        f"active_position_layout={args.active_position_layout} "
        f"short_offset={args.short_offset or 'n_sink'}"
        f" max_chunk_size={args.max_chunk_size} "
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


def run_records(
    records: list[dict],
    modes: list[str],
    loaded: PulsarModel,
    prompter: ChatMLConversation,
    args: argparse.Namespace,
) -> list[SampleResult]:
    """Every record in every mode, one progress line with a running ETA per sample."""
    results: list[SampleResult] = []
    total = len(records) * len(modes)
    done = 0
    t_start = time.perf_counter()
    for rec in records:
        for mode in modes:
            t0 = time.perf_counter()
            r = run_sample(loaded, prompter, rec, mode, args)
            results.append(r)
            done += 1
            dt = time.perf_counter() - t0
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (total - done)
            flag = r.skip_reason or ("OK" if r.correct else "x")
            print(
                f"[{done}/{total}] {mode:6s} {r.id} len={r.length} "
                f"toks={r.prompt_tokens} gen={r.gen_tokens} gold={r.gold} "
                f"pred={r.pred} demoted={r.demoted} {flag} "
                f"| {dt:.0f}s elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
            )
    return results


def report(
    results: list[SampleResult], modes: list[str], args: argparse.Namespace
) -> None:
    """Per-mode summaries, the distribution probes, the raw CoT dump and the JSON."""
    for mode in modes:
        summarize(results, mode, args.gen_tokens)
    report_histograms(results)
    report_density_by_position(results)
    if args.mode == "both":
        report_recall_advantage(results)

    dump_raw(results, args.dump_dir)
    report_skips(results, lambda r: f"{r.mode}/{r.length}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"\nwrote per-sample results -> {args.json}")


def main() -> None:
    args = parse_args()
    records = select_records(args)
    modes = ["full", "recall"] if args.mode == "both" else [args.mode]
    print_banner(args, modes, len(records))

    loaded = checkpoint.load(checkpoint.model_path(args.model), device=args.device)
    print(
        f"loaded model={args.model} archive={checkpoint.model_path(args.model)} "
        f"eos_id={loaded.config.eos_id}"
    )

    results = run_records(records, modes, loaded, ChatMLConversation(), args)
    report(results, modes, args)


if __name__ == "__main__":
    main()
