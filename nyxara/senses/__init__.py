"""NYXARA · senses/ — the doorways between her and the world (👁️👂🖐).

Perception, not tokenisation. Every module here turns something real — pixels, samples,
bytes, a page, a temperature — into a :class:`~nyxara.senses.binding.Percept` that
:class:`~nyxara.senses.binding.Binder` can hold in one frame alongside the others, so
attention and association span modalities instead of text alone.

**Seeing / hearing / reading** — :class:`~nyxara.senses.vision.Vision` and
:class:`~nyxara.senses.audio.Audio` are dependency-free analysers (PNG/BMP decode, perceptual
hashing, PCM RMS, silence segmentation, sound classification);
:class:`~nyxara.senses.nlp.NLP` does entities, sentiment, intent and readability without a
model; :class:`~nyxara.senses.ingest.Ingestor` chunks documents.

**Reaching out** — :class:`~nyxara.senses.web.WebFetcher` is the fail-closed doorway to the
live web (SSRF vetting + prompt-injection screening on every page),
:class:`~nyxara.senses.search.WebSearcher` and :class:`~nyxara.senses.browser.Browser` build
on it.

**Feeling her own substrate** — :class:`~nyxara.senses.hardware.HardwareSense` and
:class:`~nyxara.senses.system.SystemSensor` read the machine;
:class:`~nyxara.senses.thermodynamic.ThermodynamicMonitor` turns thermal and compute pressure
into an *interoceptive* signal, so load is something she feels rather than merely measures.

**Always on** — :class:`~nyxara.senses.realtime.RealtimePerception` watches and listens
between prompts, detecting speech, her name, motion and surprise natively (never the LLM) and
escalating what matters into a full sovereign cycle.
:class:`~nyxara.senses.predictive.PerceptualPredictor` scores the surprise that decides what
is worth escalating.

Both :mod:`~nyxara.senses.audio` and :mod:`~nyxara.senses.vision` define a ``parse_header``.
That name is deliberately **not** re-exported here — import it from the submodule you mean.
"""

from __future__ import annotations

from nyxara.senses.audio import (
    Audio,
    AudioAnalysis,
    AudioInfo,
    SoundProfile,
    audio_fingerprint,
    classify_sound,
    decode_pcm,
    peak,
    rms,
    silence_ratio,
    silence_segments,
    zero_crossing_rate,
)
from nyxara.senses.binding import (
    Association,
    Binder,
    Modality,
    Percept,
    PerceptualFrame,
    Provenance,
    text_simhash,
)
from nyxara.senses.browser import Browser, BrowserResult, browser_available
from nyxara.senses.generate import (
    GenerationResult,
    ImageGenerator,
    SpeechSynthesizer,
    encode_png,
    identicon_rows,
    write_png,
)
from nyxara.senses.hardware import ComputeGovernor, EventDrivenIdle, HardwareSense, Telemetry
from nyxara.senses.ingest import Chunk, Chunker, Document, DocumentType, Ingestor
from nyxara.senses.live import LiveSensor, pcm16_to_wav_bytes
from nyxara.senses.nlp import (
    NLP,
    Analysis,
    Entity,
    IntentResult,
    SentimentResult,
    detect_language,
    entities,
    intent,
    keyphrases,
    readability,
    sentences,
    sentiment,
    summarize,
    tokenize,
)
from nyxara.senses.predictive import (
    PerceptSurprise,
    PerceptualPredictor,
    SensoryPredictor,
    SurpriseResult,
    SymbolPredictor,
    SymbolSurprise,
    percept_features,
)
from nyxara.senses.realtime import ModalityChannel, PerceptEvent, RealtimePerception, detect_rects
from nyxara.senses.search import SearchResponse, SearchResult, WebSearcher
from nyxara.senses.system import DeviceSensor, NetworkSensor, SystemSensor
from nyxara.senses.thermodynamic import PrivilegedTelemetryAdapter, ThermodynamicMonitor
from nyxara.senses.vision import (
    ImageAnalysis,
    ImageInfo,
    Vision,
    average_hash,
    decode_bmp,
    decode_image,
    decode_png,
    difference_hash,
    dominant_colors,
    gray_from_pixels,
    hamming,
    hex_hash,
    perceptual_hash,
    resize_gray,
)
from nyxara.senses.watch import FileWatcher
from nyxara.senses.web import (
    FetchResult,
    HTMLExtractor,
    InjectionFlag,
    InjectionReport,
    InjectionScanner,
    Web,
    WebFetcher,
    WebPage,
)

__all__ = [
    # multimodal binding — the frame everything lands in
    "Binder", "Percept", "PerceptualFrame", "Association", "Modality", "Provenance",
    "text_simhash",
    # vision
    "Vision", "ImageAnalysis", "ImageInfo", "decode_image", "decode_png", "decode_bmp",
    "gray_from_pixels", "dominant_colors", "resize_gray", "average_hash", "difference_hash",
    "perceptual_hash", "hamming", "hex_hash",
    # hearing
    "Audio", "AudioAnalysis", "AudioInfo", "SoundProfile", "classify_sound", "decode_pcm",
    "rms", "peak", "zero_crossing_rate", "silence_segments", "silence_ratio",
    "audio_fingerprint",
    # language & documents
    "NLP", "Analysis", "Entity", "SentimentResult", "IntentResult", "tokenize", "sentences",
    "detect_language", "keyphrases", "entities", "sentiment", "intent", "summarize",
    "readability",
    "Ingestor", "Chunker", "Chunk", "Document", "DocumentType",
    # the live web (fail-closed)
    "WebFetcher", "Web", "WebPage", "FetchResult", "HTMLExtractor", "InjectionScanner",
    "InjectionReport", "InjectionFlag",
    "WebSearcher", "SearchResult", "SearchResponse",
    "Browser", "BrowserResult", "browser_available",
    # her own substrate, felt
    "HardwareSense", "Telemetry", "ComputeGovernor", "EventDrivenIdle",
    "SystemSensor", "NetworkSensor", "DeviceSensor",
    "ThermodynamicMonitor", "PrivilegedTelemetryAdapter",
    "FileWatcher",
    # always-on perception & surprise
    "RealtimePerception", "PerceptEvent", "ModalityChannel", "detect_rects",
    "LiveSensor", "pcm16_to_wav_bytes",
    "PerceptualPredictor", "PerceptSurprise", "SensoryPredictor", "SurpriseResult",
    "SymbolPredictor", "SymbolSurprise", "percept_features",
    # generation (image / speech, with real dependency-free fallbacks)
    "ImageGenerator", "SpeechSynthesizer", "GenerationResult", "encode_png", "write_png",
    "identicon_rows",
]
