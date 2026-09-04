#!/usr/bin/env python3
"""Build a corpus of real Wikipedia lead paragraphs from a dump, offline and reproducibly.

An earlier version of this read the live API. It worked and it was unusable: the API answers a
shared address from this container, a burst returns 429 for everybody on it, and at a polite pace
the run projected to over two hours and produced nothing measurable in ten minutes. A dump has no
rate limit, does not change between runs, and needs no network at test time — which matters more
than convenience, because a corpus that drifts makes every number taken against it unrepeatable.

    curl -O https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2
    python3 scripts/build_wikipedia_corpus.py enwiki-latest-pages-articles1.xml-p1p41242.bz2

What comes out is **prose**, not knowledge: the lead paragraph as Wikipedia wrote it, its title,
its page id, the categories it files itself under, and the licence. Turning it into structure is
`njp.passage`'s job and is measured separately. A builder that also decided what a paragraph meant
would make that measurement impossible to take.

The stripper is the only real work here, and it is deliberately conservative. Parentheticals,
appositives, em-dashes and long subjects are **left in**: they are what encyclopedia prose is
actually like, and cleaning them out would be quietly grading the reader on an easier corpus than
the one it claims to read.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

#: Category keyword -> the domain it stands for. A statement about *where to look*, not about what
#: will be found: the articles are whatever the encyclopedia currently files under a matching
#: category, and an article matching none is kept with an empty domain rather than dropped.
DOMAINS: Tuple[Tuple[str, str], ...] = (
    ("physics", "physics"), ("chemistry", "chemistry"), ("chemical", "chemistry"),
    ("biology", "biology"), ("biological", "biology"), ("cell ", "cell biology"),
    ("genetic", "genetics"), ("medicine", "medicine"), ("medical", "medicine"),
    ("disease", "medicine"), ("anatomy", "anatomy"), ("geology", "geology"),
    ("meteorolog", "meteorology"), ("weather", "meteorology"), ("climate", "meteorology"),
    ("astronom", "astronomy"), ("mathemat", "mathematics"), ("statistic", "statistics"),
    ("comput", "computing"), ("algorithm", "algorithms"), ("software", "computing"),
    ("engineering", "engineering"), ("electronic", "electronics"), ("metallurg", "materials"),
    ("manufactur", "manufacturing"), ("agricultur", "agriculture"), ("econom", "economics"),
    ("financ", "finance"), ("business", "business"), ("management", "business"),
    ("law", "law"), ("legal", "law"), ("government", "government"), ("politic", "politics"),
    ("sociolog", "sociology"), ("psycholog", "psychology"), ("philosoph", "philosophy"),
    ("epistemolog", "philosophy"), ("logic", "logic"), ("linguistic", "linguistics"),
    ("language", "linguistics"), ("history", "history"), ("historical", "history"),
    ("archaeolog", "archaeology"), ("geograph", "geography"), ("landform", "geography"),
    ("ecolog", "ecology"), ("zoolog", "zoology"), ("animal", "zoology"), ("botan", "botany"),
    ("plant", "botany"), ("energy", "energy"), ("transport", "transport"),
    ("architect", "architecture"), ("music", "music"), ("sport", "sport"),
    ("cooking", "cooking"), ("food", "cooking"), ("religio", "religion"), ("art", "art"),
    ("literatur", "literature"), ("military", "military"), ("war", "military"),
)

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref[^>/]*>.*?</ref>|<ref[^>]*/>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_EXTLINK = re.compile(r"\[(?:https?:|//)[^\s\]]+\s*([^\]]*)\]")
_QUOTES = re.compile(r"'{2,5}")
_HEADING = re.compile(r"^\s*={2,}.*?={2,}\s*$", re.M)
_CATEGORY = re.compile(r"\[\[Category:([^\]|]+)", re.I)
_SPACE = re.compile(r"[ \t]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_LEFTOVER = re.compile(r"[{}\[\]|]|&[a-z]+;|https?://")


def _strip_nested(text: str, open_token: str, close_token: str) -> str:
    """Remove `{{...}}` / `{|...|}` runs, honouring nesting. A regex cannot do this."""
    out: List[str] = []
    depth, index, n = 0, 0, len(text)
    while index < n:
        if text.startswith(open_token, index):
            depth += 1
            index += len(open_token)
            continue
        if depth and text.startswith(close_token, index):
            depth -= 1
            index += len(close_token)
            continue
        if not depth:
            out.append(text[index])
        index += 1
    return "".join(out)


def _links(text: str) -> str:
    """`[[a|b]]` is b, `[[a]]` is a, and a file link is nothing at all.

    Files are removed first and by scanning rather than by regex, because their captions contain
    nested links and a non-greedy `\\[\\[.*?\\]\\]` closes on the inner one, leaving `]]` behind.
    """
    out: List[str] = []
    index, n = 0, len(text)
    while index < n:
        if text.startswith("[[", index):
            close = text.find("]]", index)
            if close < 0:
                out.append(text[index:])
                break
            inner = text[index + 2:close]
            head = inner.split("|")[0].strip().lower()
            if head.startswith(("file:", "image:", "category:", "thumb")):
                index = close + 2
                continue
            out.append(inner.split("|")[-1])
            index = close + 2
            continue
        out.append(text[index])
        index += 1
    return "".join(out)


def lead_of(wikitext: str, *, sentences: int = 4) -> str:
    """The article's opening prose, as plain text."""
    text = wikitext.split("\n==", 1)[0]
    text = _COMMENT.sub(" ", text)
    text = _REF.sub(" ", text)
    text = _strip_nested(text, "{|", "|}")
    text = _strip_nested(text, "{{", "}}")
    text = _links(text)
    text = _EXTLINK.sub(r"\1", text)
    text = _TAG.sub(" ", text)
    text = _QUOTES.sub("", text)
    text = _HEADING.sub(" ", text)
    lines = [line.strip() for line in text.splitlines()]
    # A line opening on a list, table or indent marker is layout, not a sentence.
    lines = [line for line in lines if line and not line[0] in "*:;|!#"]
    joined = _SPACE.sub(" ", " ".join(lines)).strip()
    parts = _SENTENCE.split(joined)
    return " ".join(parts[:sentences]).strip()


def categories_of(wikitext: str) -> List[str]:
    return [c.strip() for c in _CATEGORY.findall(wikitext)][:12]


def domain_of(categories: Sequence[str], title: str) -> str:
    haystack = " ; ".join(categories).lower()
    for needle, domain in DOMAINS:
        if needle in haystack:
            return domain
    return ""


def pages(path: Path) -> Iterator[Tuple[str, int, str]]:
    """Stream `(title, pageid, wikitext)` for every real article in the dump."""
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:  # type: ignore
        title, pageid, ns, redirect, text = "", 0, -1, False, None
        for event, element in ElementTree.iterparse(handle, events=("start", "end")):
            tag = element.tag.rsplit("}", 1)[-1]
            if event == "start" and tag == "page":
                title, pageid, ns, redirect, text = "", 0, -1, False, None
                continue
            if event != "end":
                continue
            if tag == "title":
                title = element.text or ""
            elif tag == "ns":
                ns = int(element.text or -1)
            elif tag == "redirect":
                redirect = True
            elif tag == "id" and not pageid:
                pageid = int(element.text or 0)
            elif tag == "text":
                text = element.text or ""
            elif tag == "page":
                if ns == 0 and not redirect and text:
                    yield title, pageid, text
                element.clear()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--out", default="nyxara/njp/data/wikipedia_leads.jsonl.gz")
    parser.add_argument("--per-domain", type=int, default=80)
    parser.add_argument("--min-words", type=int, default=25)
    parser.add_argument("--licence", default="CC BY-SA 4.0")
    args = parser.parse_args(argv)

    kept: Dict[str, int] = {}
    rows: List[Dict] = []
    seen = skipped = 0
    for title, pageid, wikitext in pages(Path(args.dump)):
        seen += 1
        low = title.lower()
        if low.startswith(("list of", "index of", "outline of", "glossary of", "timeline of",
                           "comparison of", "bibliography of")):
            continue
        categories = categories_of(wikitext)
        domain = domain_of(categories, title)
        if not domain or kept.get(domain, 0) >= args.per_domain:
            continue
        lead = lead_of(wikitext)
        if len(lead.split()) < args.min_words or lead.count(".") < 2:
            skipped += 1
            continue
        # A lead still carrying markup is a stripper failure, and letting it through would put
        # the stripper's defects into the reader's score.
        if _LEFTOVER.search(lead):
            skipped += 1
            continue
        kept[domain] = kept.get(domain, 0) + 1
        rows.append({"title": title, "pageid": pageid, "text": lead,
                     "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                     "domain": domain, "category": categories[0] if categories else "",
                     "fetched": time.strftime("%Y-%m-%d", time.gmtime()),
                     "licence": args.licence, "source": "en.wikipedia.org"})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (r["domain"], r["title"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{seen} pages seen, {skipped} leads rejected by the quality filter, "
          f"{len(rows)} kept across {len(kept)} domains", file=sys.stderr)
    for domain, count in sorted(kept.items()):
        print(f"  {domain:<16}{count:>5}", file=sys.stderr)
    print(f"-> {out} ({out.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
