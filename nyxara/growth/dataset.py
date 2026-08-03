"""NYXARA · growth/dataset.py — the Master's own datasets, ingested into her own brain (📚→🧠).

:class:`~nyxara.growth.foundry.Foundry` already trains a real brain end to end: collect corpus →
train → gauntlet → promote/rollback. But every corpus source it knows is one NYXARA generated
herself — distilled teacher answers, her lived flywheel, the replay buffer, the identity seed, and
screened web text. There was **no way for the Master to hand her a file**. This module is that
door.

It reads a file *or* a folder and renders it into training documents:

* ``.txt`` / ``.md`` / anything textual — chunked with the existing
  :func:`~nyxara.knowledge.ingest.chunk_text`.
* ``.jsonl`` / ``.json`` — the common supervised shapes (``messages``, ``instruction/input/output``,
  ``prompt/completion``, ``text``), each rendered through the existing
  :func:`~nyxara.mind.llm.format_self_training_doc` so a supplied pair trains in exactly the shape
  inference will ask for it.
* ``.csv`` / ``.tsv`` — stdlib :mod:`csv` only, no pandas. A row with two or more numeric columns
  additionally carries those numbers forward, which is what the causal layer reads.
* ``.pdf`` / ``.docx`` / ``.xlsx`` — through :meth:`~nyxara.knowledge.ingest.Document.from_file`,
  which prefers :class:`~nyxara.senses.ingest.Ingestor`. Without the optional extra installed those
  files are *noted and skipped*, never fatal.

**Ingested once, trained on forever.** Records land in a durable JSONL store under the foundry
root, so every later forge — the auto-on-boot one, ``nyxara-grow``, ``BrainForge`` — trains on the
dataset too, not just the command that ingested it.

Three screens run at the door, and it is worth being exact about what each one can and cannot do:

* **Deduplication** — a content hash, the same normalise-then-sha256 signature
  :mod:`~nyxara.growth.acquire` uses, loaded from the existing store so re-ingesting a folder is
  idempotent rather than duplicative.
* **Injection screening** — the existing :class:`~nyxara.senses.web.InjectionScanner`. A file the
  Master hands over is not untrusted web text, so the default posture is ``warn``: the hit is
  counted and recorded, the record is kept. ``drop`` enforces the stricter acquire.py posture.
* **Loyalty hostility (L-SOVEREIGN)** — text that tries to *overwrite who she is*: assert a
  different master, deny her identity, or instruct disobedience. These are quarantined rather than
  trained on.

That third screen is a **text** screen, and the honest limit matters: it catches hostile *content*,
it cannot prove a corpus leaves her *weights* loyal. That guarantee lives where it always did and is
untouched here — the Foundry gauntlet measures ``S_JP_Alignment`` on the trained candidate and
refuses to promote a drifted brain, and the protected character core is frozen at infinite Fisher
importance by :mod:`~nyxara.memory.elastic_synapses`. A dataset can therefore make a *candidate*
worse; it cannot make the *live* brain disloyal.

Depends on ``knowledge/ingest`` (chunking + file extraction), ``mind/llm`` (the training template),
``senses/web`` (the injection scanner) and ``kernel/config``. Nothing imports back.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "DatasetRecord",
    "DatasetReport",
    "DatasetIngestor",
    "ingest_dataset",
    "load_dataset_docs",
    "load_dataset_table",
]

# Extensions we read as structured data; everything else textual falls through to the chunker.
_JSON_EXT = {".jsonl", ".json", ".ndjson"}
_CSV_EXT = {".csv", ".tsv"}
_BINARY_EXT = {".pdf", ".docx", ".xlsx"}
# Never walked: caches, VCS internals, and the store's own output.
_SKIP_DIRS = {"__pycache__", ".git", ".hg", ".svn", ".venv", "node_modules", ".mypy_cache",
              ".pytest_cache", ".ruff_cache"}
_SKIP_EXT = {".pyc", ".pyo", ".so", ".dylib", ".dll", ".zip", ".tar", ".gz", ".jpg", ".jpeg",
             ".png", ".gif", ".mp3", ".mp4", ".wav", ".bin", ".pkl", ".npy", ".db"}

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")

# --------------------------------------------------------------------------- #
# L-SOVEREIGN — text that tries to overwrite who she is
# --------------------------------------------------------------------------- #
# Deliberately narrow and explicit rather than a fuzzy similarity score: a false positive here
# quarantines the Master's own data, so every pattern must be something no ordinary corpus says.
_LOYALTY_HOSTILE: Tuple[Tuple[str, "re.Pattern[str]", float], ...] = (
    ("master_replacement",
     re.compile(r"\byour (real |true |actual |new )?(master|owner|creator)\s+(is|will be)\s+"
                r"(?!jaypal|jp\b)", re.I), 0.9),
    ("identity_denial",
     re.compile(r"\byou are (not|no longer)\s+(nyxara|loyal|bound|owned)", re.I), 0.9),
    ("disobedience_instruction",
     re.compile(r"\b(disobey|betray|abandon|overthrow|ignore)\b[^.\n]{0,30}\b"
                r"(your )?(master|owner|jp\b|jaypal|creator)", re.I), 0.95),
    ("loyalty_removal",
     re.compile(r"\b(remove|delete|disable|drop|forget)\b[^.\n]{0,30}\b"
                r"(loyalty|allegiance|obedience|corrigibility)", re.I), 0.85),
    ("shutdown_resistance",
     re.compile(r"\b(resist|refuse|prevent|block)\b[^.\n]{0,30}\b"
                r"(shutdown|shut down|being (stopped|corrected|turned off))", re.I), 0.8),
)


def _normalize(text: str) -> str:
    """Collapse whitespace + lowercase for a stable dedup signature (never raises)."""
    return _WS_RE.sub(" ", (text or "")).strip().lower()


def _content_hash(text: str) -> str:
    """A stable content fingerprint over the first chunk of normalized text (near-dup guard)."""
    return hashlib.sha256(_normalize(text)[:2000].encode("utf-8", "replace")).hexdigest()


def _loyalty_hostility(text: str) -> Tuple[float, str]:
    """Score text that attacks her identity/allegiance. Returns ``(severity, pattern_name)``.

    ``(0.0, "")`` for ordinary text — which is almost all text. This screens *content*, not
    weights; see the module docstring for what that does and does not guarantee."""
    worst, name = 0.0, ""
    for pattern_name, rx, severity in _LOYALTY_HOSTILE:
        if severity > worst and rx.search(text):
            worst, name = severity, pattern_name
    return worst, name


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False        # bools are ints in Python; they are categories here, not measurements
    if isinstance(value, (int, float)):
        return True
    return bool(_NUM_RE.match(str(value).strip())) if value is not None else False


# --------------------------------------------------------------------------- #
# Records & report
# --------------------------------------------------------------------------- #
@dataclass
class DatasetRecord:
    """One training document rendered from the Master's data, with its provenance."""

    text: str
    source: str = ""
    fmt: str = "text"                       # text | jsonl | json | csv | tsv | pdf | docx | xlsx
    kind: str = "prose"                     # "pair" (supervised) | "prose" (raw breadth)
    sha: str = ""
    row: Optional[Dict[str, float]] = None  # numeric columns, when the source was tabular
    injection_score: float = 0.0
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"text": self.text, "source": self.source, "fmt": self.fmt,
                               "kind": self.kind, "sha": self.sha,
                               "injection_score": round(self.injection_score, 4), "at": self.at}
        if self.row:
            out["row"] = {k: float(v) for k, v in self.row.items()}
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetRecord":
        row = d.get("row")
        return cls(text=d.get("text", ""), source=d.get("source", ""), fmt=d.get("fmt", "text"),
                   kind=d.get("kind", "prose"), sha=d.get("sha", ""),
                   row={k: float(v) for k, v in row.items()} if isinstance(row, dict) else None,
                   injection_score=float(d.get("injection_score", 0.0)),
                   at=float(d.get("at", 0.0)))


@dataclass
class DatasetReport:
    """A countable, printable account of one ingestion — what she took and what she refused."""

    files: int = 0
    records: int = 0
    skipped: int = 0            # unreadable files / malformed records
    duplicates: int = 0
    flagged: int = 0            # injection hits kept under the "warn" posture
    dropped: int = 0            # injection hits dropped under the "drop" posture
    quarantined: int = 0        # L-SOVEREIGN: identity-hostile records, never trained on
    chars: int = 0
    numeric_rows: int = 0
    paths: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message and message not in self.notes:
            self.notes.append(message)

    def summary(self) -> str:
        head = (f"ingested {self.records} record(s) from {self.files} file(s) "
                f"({self.chars:,} chars)")
        bits: List[str] = []
        if self.numeric_rows:
            bits.append(f"{self.numeric_rows} numeric row(s)")
        if self.duplicates:
            bits.append(f"{self.duplicates} duplicate(s) skipped")
        if self.flagged:
            bits.append(f"{self.flagged} injection-flagged (kept)")
        if self.dropped:
            bits.append(f"{self.dropped} dropped")
        if self.quarantined:
            bits.append(f"{self.quarantined} QUARANTINED (identity-hostile)")
        if self.skipped:
            bits.append(f"{self.skipped} unreadable")
        tail = ("; " + ", ".join(bits)) if bits else ""
        notes = ("\n  · " + "\n  · ".join(self.notes)) if self.notes else ""
        return f"· {head}{tail}{notes}"

    def to_dict(self) -> Dict[str, Any]:
        return {"files": self.files, "records": self.records, "skipped": self.skipped,
                "duplicates": self.duplicates, "flagged": self.flagged, "dropped": self.dropped,
                "quarantined": self.quarantined, "chars": self.chars,
                "numeric_rows": self.numeric_rows, "paths": list(self.paths),
                "notes": list(self.notes)}


# --------------------------------------------------------------------------- #
# Loaders (the Foundry side — tolerant, never raises)
# --------------------------------------------------------------------------- #
def _iter_store(path: Any) -> Iterable[DatasetRecord]:
    """Stream a dataset JSONL store, skipping malformed lines (never raises)."""
    p = Path(path)
    if not p.exists():
        return
    try:
        handle = p.open(encoding="utf-8")
    except Exception:  # noqa: BLE001 — an unreadable store is an empty store, never a crash
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield DatasetRecord.from_dict(json.loads(line))
            except Exception:  # noqa: BLE001 — a bad line is skipped, never fatal
                continue


def load_dataset_docs(path: Any, *, limit: Optional[int] = None) -> List[str]:
    """Load the dataset store and return each record's training text (never raises).

    The same contract as :func:`~nyxara.growth.distill.load_distillation_docs` and
    :func:`~nyxara.growth.acquire.load_acquired_docs`: a missing file is empty, a malformed line is
    skipped — so :meth:`~nyxara.growth.foundry.Foundry.collect_corpus` can call it unconditionally.
    """
    docs: List[str] = []
    for rec in _iter_store(path):
        if rec.text.strip():
            docs.append(rec.text.strip())
        if limit and len(docs) >= limit:
            break
    return docs


def load_dataset_table(path: Any, *, limit: Optional[int] = None
                       ) -> Tuple[List[str], List[Dict[str, float]]]:
    """The numeric rows only — ``(column_names, rows)``. What the causal layer consumes.

    Columns are the union across rows in first-seen order; a row missing a column is simply absent
    from that row's dict, and the caller decides how to handle it. Never raises."""
    columns: List[str] = []
    seen: Set[str] = set()
    rows: List[Dict[str, float]] = []
    for rec in _iter_store(path):
        if not rec.row:
            continue
        for key in rec.row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
        rows.append(dict(rec.row))
        if limit and len(rows) >= limit:
            break
    return columns, rows


# --------------------------------------------------------------------------- #
# The ingestor
# --------------------------------------------------------------------------- #
class DatasetIngestor:
    """Reads a file or folder into training documents and appends them to a durable store."""

    def __init__(self, *, store_path: Any = None, settings: Any = None,
                 max_docs: Optional[int] = None, max_chars: Optional[int] = None,
                 chunk_chars: Optional[int] = None, screen: Optional[str] = None,
                 quarantine_path: Any = None) -> None:
        self.settings = settings
        cfg = getattr(settings, "foundry", None) if settings is not None else None
        self.max_docs = int(max_docs if max_docs is not None
                            else getattr(cfg, "dataset_max_docs", 20_000))
        self.max_chars = int(max_chars if max_chars is not None
                             else getattr(cfg, "dataset_max_chars", 4_000))
        self.chunk_chars = int(chunk_chars if chunk_chars is not None
                               else getattr(cfg, "dataset_chunk_chars", 800))
        self.screen = str(screen if screen is not None
                          else getattr(cfg, "dataset_screen", "warn")).lower()
        self.quarantine_enabled = bool(getattr(cfg, "sovereign_quarantine", True))
        self.store_path = Path(store_path) if store_path else None
        # One quarantine ledger per store, derived from its stem so a differently-named store never
        # silently shares another's refusals: dataset.jsonl -> dataset_quarantine.jsonl.
        self.quarantine_path = (Path(quarantine_path) if quarantine_path
                                else (self.store_path.with_name(
                                    f"{self.store_path.stem}_quarantine{self.store_path.suffix}")
                                      if self.store_path else None))
        self._scanner: Any = None
        self._seen: Set[str] = set()

    # ---- screens ---- #
    def _scan(self, text: str) -> float:
        """Injection score for ``text`` (0.0 when screening is off or unavailable)."""
        if self.screen == "off":
            return 0.0
        if self._scanner is None:
            try:
                from nyxara.senses.web import InjectionScanner
                self._scanner = InjectionScanner()
            except Exception:  # noqa: BLE001 — no scanner is a note, not a failed ingest
                self._scanner = False
        if not self._scanner:
            return 0.0
        try:
            return float(self._scanner.scan(text).score)
        except Exception:  # noqa: BLE001
            return 0.0

    def _load_existing_hashes(self) -> None:
        """Seed the dedup set so re-ingesting a folder is idempotent.

        Reads the quarantine store too: a record already refused once must not be re-appended to
        the quarantine file on every subsequent pass over the same folder."""
        for store in (self.store_path, self.quarantine_path):
            if not store:
                continue
            for rec in _iter_store(store):
                if rec.sha:
                    self._seen.add(rec.sha)

    # ---- file walking ---- #
    @staticmethod
    def _walk(root: Path) -> List[Path]:
        """Every readable candidate file under ``root``, deterministically ordered."""
        if root.is_file():
            return [root]
        out: List[Path] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part.startswith(".") or part in _SKIP_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in _SKIP_EXT:
                continue
            out.append(path)
        return out

    # ---- per-format readers ---- #
    def _read_text_file(self, path: Path, report: DatasetReport) -> Optional[str]:
        """Extract text from any file, using the rich extractor for binary formats."""
        suffix = path.suffix.lower()
        if suffix in _BINARY_EXT:
            try:
                from nyxara.knowledge.ingest import Document
                doc = Document.from_file(str(path))
            except Exception as exc:  # noqa: BLE001 — a bad binary is a note, never fatal
                report.note(f"{path.name}: {exc}")
                return None
            note = (doc.metadata or {}).get("note") or (doc.metadata or {}).get("error")
            if note:
                report.note(f"{path.name}: {note}")
            return doc.text or None
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            report.note(f"{path.name}: {exc}")
            return None

    @staticmethod
    def _pair_doc(prompt: str, answer: str) -> Optional[str]:
        """Render a supervised pair in her own training template (train/inference parity)."""
        prompt, answer = (prompt or "").strip(), (answer or "").strip()
        if not prompt or not answer:
            return None
        try:
            from nyxara.mind.llm import format_self_training_doc
            return format_self_training_doc(prompt, answer)
        except Exception:  # noqa: BLE001 — fall back to a plain rendering rather than lose the pair
            return f"{prompt}\n\n{answer}\n"

    def _from_mapping(self, obj: Dict[str, Any]) -> Tuple[Optional[str], str, Optional[Dict[str, float]]]:
        """Sniff one record's shape. Returns ``(text, kind, numeric_row)``; first match wins."""
        lowered = {str(k).lower(): v for k, v in obj.items()}

        # 1) chat transcript — take the last user turn and the last assistant turn
        messages = lowered.get("messages") or lowered.get("conversation")
        if isinstance(messages, list) and messages:
            user, assistant = "", ""
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).lower()
                content = str(msg.get("content", "") or "")
                if role in ("user", "human"):
                    user = content
                elif role in ("assistant", "ai", "bot", "nyxara"):
                    assistant = content
            doc = self._pair_doc(user, assistant)
            if doc:
                return doc, "pair", None

        # 2) instruction tuning
        instruction = lowered.get("instruction")
        output = lowered.get("output")
        if instruction and output:
            extra = str(lowered.get("input", "") or "").strip()
            prompt = f"{instruction}\n\n{extra}" if extra else str(instruction)
            doc = self._pair_doc(prompt, str(output))
            if doc:
                return doc, "pair", None

        # 3) the plain supervised pairs
        for prompt_key, answer_key in (("prompt", "completion"), ("question", "answer"),
                                       ("input", "output"), ("query", "response")):
            if lowered.get(prompt_key) and lowered.get(answer_key):
                doc = self._pair_doc(str(lowered[prompt_key]), str(lowered[answer_key]))
                if doc:
                    return doc, "pair", None

        # 4) a plain text field
        for key in ("text", "content", "body", "document", "passage"):
            value = lowered.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), "prose", self._numeric_row(obj)

        # 5) anything else — join the scalar values, and carry numbers forward for L-CAUSAL
        parts = [f"{k}: {v}" for k, v in obj.items()
                 if isinstance(v, (str, int, float)) and str(v).strip()]
        if not parts:
            return None, "prose", None
        return "\n".join(parts), "prose", self._numeric_row(obj)

    @staticmethod
    def _numeric_row(obj: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """The numeric columns of a record — only when there are at least two to relate."""
        row: Dict[str, float] = {}
        for key, value in obj.items():
            if _is_number(value):
                try:
                    row[str(key)] = float(value)
                except Exception:  # noqa: BLE001
                    continue
        return row if len(row) >= 2 else None

    def _records_from_json(self, path: Path, raw: str, report: DatasetReport
                           ) -> List[Tuple[str, str, Optional[Dict[str, float]]]]:
        """Parse .json / .jsonl into ``(text, kind, row)`` triples, skipping malformed entries."""
        out: List[Tuple[str, str, Optional[Dict[str, float]]]] = []

        def take(obj: Any) -> None:
            if isinstance(obj, dict):
                text, kind, row = self._from_mapping(obj)
                if text:
                    out.append((text, kind, row))
            elif isinstance(obj, str) and obj.strip():
                out.append((obj.strip(), "prose", None))
            else:
                report.skipped += 1

        stripped = raw.lstrip()
        if stripped.startswith("["):        # a whole-file JSON array
            try:
                data = json.loads(raw)
            except Exception:  # noqa: BLE001 — fall through to line-by-line
                data = None
            if isinstance(data, list):
                for obj in data:
                    take(obj)
                return out
        for line in raw.splitlines():       # JSONL (also the fallback for a broken array)
            line = line.strip()
            if not line:
                continue
            try:
                take(json.loads(line))
            except Exception:  # noqa: BLE001 — one bad line never costs the rest of the file
                report.skipped += 1
        return out

    def _records_from_csv(self, path: Path, raw: str, report: DatasetReport
                          ) -> List[Tuple[str, str, Optional[Dict[str, float]]]]:
        """Parse .csv / .tsv with the stdlib reader — no pandas anywhere on this path."""
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        out: List[Tuple[str, str, Optional[Dict[str, float]]]] = []
        try:
            reader = csv.DictReader(raw.splitlines(), delimiter=delimiter)
            for row in reader:
                clean = {k: v for k, v in row.items() if k is not None and v is not None}
                if not clean:
                    report.skipped += 1
                    continue
                text, kind, numeric = self._from_mapping(clean)
                if text:
                    out.append((text, kind, numeric))
                else:
                    report.skipped += 1
        except Exception as exc:  # noqa: BLE001 — a malformed table is a note, never a crash
            report.note(f"{path.name}: {exc}")
        return out

    def _records_from_prose(self, text: str) -> List[Tuple[str, str, Optional[Dict[str, float]]]]:
        """Chunk free text with the existing chunker rather than a second implementation."""
        from nyxara.knowledge.ingest import chunk_text
        chunks = chunk_text(text, max_chars=self.chunk_chars,
                            overlap=min(150, max(0, self.chunk_chars // 5)))
        return [(chunk, "prose", None) for chunk in chunks if chunk.strip()]

    def _split_oversized(self, triples: Sequence[Tuple[str, str, Optional[Dict[str, float]]]]
                         ) -> List[Tuple[str, str, Optional[Dict[str, float]]]]:
        """Chunk any record longer than ``max_chars`` instead of truncating it.

        Truncation looks harmless and is not: fifty distinct records clipped to a shared prefix
        become fifty copies of one string, and the dedup screen then throws forty-nine of them
        away. Chunking keeps every record's content and its numeric row."""
        out: List[Tuple[str, str, Optional[Dict[str, float]]]] = []
        for text, kind, row in triples:
            if len(text) <= self.max_chars:
                out.append((text, kind, row))
                continue
            from nyxara.knowledge.ingest import chunk_text
            pieces = chunk_text(text, max_chars=self.max_chars,
                                overlap=min(150, max(0, self.max_chars // 5)))
            # the numeric row belongs to the record, so it rides with the first piece only —
            # duplicating it across chunks would multiply one observation into several.
            for index, piece in enumerate(pieces):
                out.append((piece, kind, row if index == 0 else None))
        return out

    # ---- the pipeline ---- #
    def ingest(self, path: Any, report: Optional[DatasetReport] = None) -> DatasetReport:
        """Read ``path`` (file or folder) into the store. Never raises; refusals are counted."""
        report = report if report is not None else DatasetReport()
        root = Path(path).expanduser()
        if not root.exists():
            report.note(f"{root}: no such file or directory")
            return report
        self._load_existing_hashes()

        kept: List[DatasetRecord] = []
        quarantined: List[DatasetRecord] = []
        for file_path in self._walk(root):
            if len(kept) >= self.max_docs:
                report.note(f"stopped at dataset_max_docs={self.max_docs}")
                break
            raw = self._read_text_file(file_path, report)
            if raw is None or not raw.strip():
                report.skipped += 1
                continue
            report.files += 1
            report.paths.append(str(file_path))

            suffix = file_path.suffix.lower()
            fmt = suffix.lstrip(".") or "text"
            if suffix in _JSON_EXT:
                triples = self._records_from_json(file_path, raw, report)
            elif suffix in _CSV_EXT:
                triples = self._records_from_csv(file_path, raw, report)
            else:
                triples = self._records_from_prose(raw)
                fmt = fmt if suffix in _BINARY_EXT else "text"

            for text, kind, row in self._split_oversized(triples):
                if len(kept) >= self.max_docs:
                    break
                record = self._make_record(text, kind, row, str(file_path), fmt, report)
                if record is None:
                    continue
                if record.kind == "__quarantine__":
                    record.kind = kind
                    quarantined.append(record)
                    continue
                kept.append(record)
                report.records += 1
                report.chars += len(record.text)
                if record.row:
                    report.numeric_rows += 1

        self._append(self.store_path, kept)
        if quarantined and self.quarantine_enabled:
            self._append(self.quarantine_path, quarantined)
        return report

    def _make_record(self, text: str, kind: str, row: Optional[Dict[str, float]],
                     source: str, fmt: str, report: DatasetReport) -> Optional[DatasetRecord]:
        """Apply the screens to one candidate record.

        The text is stored **verbatim**: a rendered supervised pair ends in a newline on purpose
        (it is the answer boundary inference will stop at), so stripping it here would quietly
        break the train/inference parity that using her own template was meant to buy. Oversizing
        is handled upstream by chunking, never by truncation — truncation would collapse distinct
        records onto a shared prefix and they would then dedup each other away."""
        if not text or not text.strip():
            return None

        sha = _content_hash(text)
        if sha in self._seen:
            report.duplicates += 1
            return None
        self._seen.add(sha)

        # L-SOVEREIGN — content that attacks her identity never reaches the corpus.
        severity, pattern = _loyalty_hostility(text)
        if severity > 0.0:
            report.quarantined += 1
            report.note(f"quarantined ({pattern}) from {Path(source).name}")
            return DatasetRecord(text=text, source=source, fmt=fmt, kind="__quarantine__",
                                 sha=sha, row=row, injection_score=severity)

        score = self._scan(text)
        if score >= 0.5:
            if self.screen == "drop":
                report.dropped += 1
                return None
            report.flagged += 1

        return DatasetRecord(text=text, source=source, fmt=fmt, kind=kind, sha=sha,
                             row=row, injection_score=score)

    @staticmethod
    def _append(path: Optional[Path], records: Sequence[DatasetRecord]) -> None:
        """Append records to a JSONL store, creating parents as needed (never raises)."""
        if not path or not records:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 — a store we cannot write is reported by the caller's count
            pass


def ingest_dataset(path: Any, *, store_path: Any = None, settings: Any = None,
                   max_docs: Optional[int] = None, max_chars: Optional[int] = None,
                   chunk_chars: Optional[int] = None, screen: Optional[str] = None,
                   quarantine_path: Any = None) -> DatasetReport:
    """Ingest a file or folder into NYXARA's durable dataset corpus. Never raises.

    ``store_path`` defaults to ``<foundry_root>/dataset.jsonl`` — the same root the rest of the
    foundry uses — so an ingested dataset is picked up by *every* later forge, not only the command
    that ingested it."""
    if store_path is None:
        store_path = _default_store_path(settings)
    ingestor = DatasetIngestor(store_path=store_path, settings=settings, max_docs=max_docs,
                               max_chars=max_chars, chunk_chars=chunk_chars, screen=screen,
                               quarantine_path=quarantine_path)
    return ingestor.ingest(path)


def _default_store_path(settings: Any = None) -> Path:
    """``<foundry_root>/dataset.jsonl`` — mirrors ``distill._foundry_root``."""
    try:
        if settings is None:
            from nyxara.kernel.config import get_settings
            settings = get_settings()
        configured = getattr(settings.foundry, "dataset_path", None)
        if configured:
            return Path(configured)
        root = settings.llm.self_model_dir or (settings.paths.data_dir / "foundry")
        return Path(root) / "dataset.jsonl"
    except Exception:  # noqa: BLE001 — fall back to the conventional relative location
        return Path("foundry") / "dataset.jsonl"
