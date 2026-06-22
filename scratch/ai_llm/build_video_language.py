#!/usr/bin/env python3
"""Build the per-video **language channel** artifact for the V3 ABR variant.

What V3 needs
-------------
V3 = V2 + a content-derived natural-language string per video, embedded by the frozen LLM's
own token embeddings and concatenated with the numeric soft tokens (see model.py / train.py).
The string must be **world-knowledge / open-vocabulary** content (genre, title, content tags) —
NOT the five numeric scalars re-spelled (the "redundancy rule", docs/benchmark-plan.md §4.5).
The abr `video_catalog.yaml` already carries exactly this: a human `title`, content `tags`, and a
coarse `genre` (reused from split_dataset_videos.genre_for) — deterministic, cheap, reproducible,
no VLM. This script templates one string per video and writes `config/video_semantics.json`.

Tokenization is intentionally **deferred to load time** (train.py / evaluate.py run the
`AutoTokenizer(model_id)` for variant==v3), so this file stores only `{genre, text}` plus
provenance. That keeps the artifact tokenizer-agnostic and the dataset rebuild-free.

Usage:
    python build_video_language.py                       # -> config/video_semantics.json
    python build_video_language.py --out config/video_semantics.json --catalog <path>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

try:
    from ai_llm.split_dataset_videos import DEFAULT_CATALOG, genre_for
    from ai_llm.model import DEFAULT_MODEL_ID
    from ai_llm.paths import CONFIG_DIR
except ImportError:  # run directly from scratch/ai_llm
    from split_dataset_videos import DEFAULT_CATALOG, genre_for
    from model import DEFAULT_MODEL_ID
    from paths import CONFIG_DIR

# Boilerplate tags that carry no content signal (encoding/source/quality/provenance housekeeping).
# Dropped so the language string stays world-knowledge about the scene, not catalog plumbing.
BOILERPLATE_TAGS = {
    "pexels", "4k", "8k", "hd", "uhd", "benchmark", "standard", "tier", "video", "discovered",
}
# Query terms that are resolution/housekeeping, not content (the notes `query:` for discovered clips).
BOILERPLATE_QUERY_WORDS = {"4k", "8k", "hd", "uhd", "video", "footage", "clip"}
# Titles auto-generated for discovered Pexels clips carry no content ("Pexels video 30935167").
_BOILERPLATE_TITLE_RE = re.compile(r"^pexels video \d+$", re.IGNORECASE)

# The templating rule is pinned in provenance so the artifact is reproducible/auditable.
TEMPLATE = "Content: {genre} scene. {scene}. Tags: {tags}."


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug_phrase(entry: Dict) -> str:
    """Human scene description mined from the Pexels page_url slug.

    e.g. ``.../video/aerial-view-of-car-passing-track-and-trees-30935167/`` ->
    "aerial view of car passing track and trees". The trailing numeric id is stripped.
    Returns "" when no usable slug is present.
    """
    url = ((entry.get("download") or {}).get("page_url") or "").strip().lower()
    m = re.search(r"/video/([a-z0-9\-]+?)(?:-\d+)?/?$", url)
    if not m:
        return ""
    words = [w for w in m.group(1).split("-") if w and not w.isdigit()]
    return " ".join(words)


def _query_terms(entry: Dict) -> List[str]:
    """Content words from the discovery query in ``notes`` (e.g. 'query: 4k sports running')."""
    notes = (entry.get("notes") or "").lower()
    m = re.search(r"query:\s*([^)\']+)", notes)
    if not m:
        return []
    out: List[str] = []
    for w in m.group(1).strip().split():
        w = w.strip(".,")
        if w and w not in BOILERPLATE_QUERY_WORDS:
            out.append(w)
    return out


def _scene_phrase(entry: Dict, genre: str) -> str:
    """Best human scene description: real title, else page_url slug, else the genre."""
    title = str(entry.get("title") or "").strip()
    if title and not _BOILERPLATE_TITLE_RE.match(title):
        return title
    slug = _slug_phrase(entry)
    if slug:
        return slug.capitalize()
    return "%s footage" % genre


def _content_tags(entry: Dict, genre: str) -> List[str]:
    """De-duplicated content tags: catalog tags ∪ discovery-query terms, boilerplate removed.

    Order preserved (catalog tags first, then query terms). Falls back to the genre when the
    catalog carries only boilerplate (common for auto-discovered clips tagged just 'discovered').
    """
    seen = set()
    out: List[str] = []
    for tag in list(entry.get("tags") or []) + _query_terms(entry):
        t = str(tag).strip().lower()
        if not t or t in BOILERPLATE_TAGS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out or [genre]


def build_text(entry: Dict, genre: str) -> str:
    """Template one natural-language content string for a catalog entry.

    Rich-title example: "Content: nature scene. Scenic Hiking Adventure in Mountain Forest.
    Tags: nature, hiking." Discovered-clip example (title is boilerplate, mine the URL slug +
    query): "Content: sports scene. Aerial view of car passing track and trees. Tags: sports,
    running."
    """
    scene = _scene_phrase(entry, genre)
    tags = _content_tags(entry, genre)
    return TEMPLATE.format(genre=genre, scene=scene, tags=", ".join(tags))


def build_semantics(catalog_path: str) -> Dict[str, Dict[str, str]]:
    import yaml  # provided by the project .venv

    with open(catalog_path, "r", encoding="utf-8") as fh:
        cat = yaml.safe_load(fh)
    entries = cat.get("videos", cat) if isinstance(cat, dict) else cat
    videos: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        vid = entry.get("id")
        if not vid:
            continue
        genre = genre_for(entry)
        videos[vid] = {"genre": genre, "text": build_text(entry, genre)}
    return videos


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--catalog", default=DEFAULT_CATALOG, help="abr video_catalog.yaml.")
    ap.add_argument(
        "--out",
        default=str(Path(CONFIG_DIR) / "video_semantics.json"),
        help="Output JSON (default: config/video_semantics.json).",
    )
    ap.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Tokenizer/LLM the loader will embed this text with (recorded for provenance).",
    )
    args = ap.parse_args()

    videos = build_semantics(args.catalog)
    if not videos:
        raise SystemExit("No video entries found in catalog %s" % args.catalog)

    artifact = {
        "_provenance": {
            "catalog": args.catalog,
            "catalog_sha256": _file_sha256(args.catalog),
            "template": TEMPLATE,
            "boilerplate_tags_dropped": sorted(BOILERPLATE_TAGS),
            "model_id": args.model_id,
            "generated_utc": _dt.datetime.utcnow().isoformat() + "Z",
            "n_videos": len(videos),
        },
        "videos": videos,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True)

    print("Wrote %d video language strings -> %s" % (len(videos), out_path))
    # Show a couple of examples + per-genre counts so the templating is eyeballable.
    from collections import Counter

    genres = Counter(v["genre"] for v in videos.values())
    print("genres: %s" % dict(sorted(genres.items())))
    for vid in list(videos)[:3]:
        print("  %-22s %s" % (vid, videos[vid]["text"]))


if __name__ == "__main__":
    main()
