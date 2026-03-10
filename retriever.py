"""
retriever.py — Knowledge-base RAG for Naya Mitra AI
─────────────────────────────────────────────────────
Loads all four sacred-text datasets into memory at startup.
retrieve(query, top_k) returns the most relevant passages via TF-IDF cosine
similarity so we never fabricate verses.
"""

import json
import math
import re
from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).parent / "data"

# ── Passage data class ─────────────────────────────────────────────────────────

class Passage:
    __slots__ = ("source", "ref", "sanskrit", "transliteration", "text", "_tokens")

    def __init__(self, source, ref, text, sanskrit="", transliteration=""):
        self.source = source          # "Bhagavad Gita" | "Hitopadesha" | …
        self.ref = ref                # "Chapter 2, Verse 47" | "Story: Tiger…"
        self.text = text              # English translation / teaching
        self.sanskrit = sanskrit
        self.transliteration = transliteration
        self._tokens: dict = {}

    def search_text(self) -> str:
        return (self.text + " " + self.ref).lower()


# ── Tokeniser ──────────────────────────────────────────────────────────────────

_STOP = {
    "a","an","the","is","it","in","of","to","and","or","for","with","that",
    "this","as","be","by","at","from","on","are","was","were","not","but",
    "have","has","had","do","does","did","i","you","he","she","we","they",
    "me","him","her","us","them","my","your","his","our","their","what","how",
    "when","where","who","which","one","if","so","all","will","can","would",
    "could","should","also","more","very","just","said","say","see","know",
    "get","go","come","make","take","give","well","even","after","into"
}

def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return [w for w in words if w not in _STOP and len(w) > 2]


def _tfidf_scores(query_tokens: list[str], corpus: list[Passage]) -> list[float]:
    """Return a cosine-like TF-IDF score for each passage against the query."""
    N = len(corpus)
    if N == 0:
        return []

    # Build IDF (document frequency)
    df: dict[str, int] = {}
    for p in corpus:
        if not p._tokens:
            p._tokens = {}
            tokens = _tokenize(p.search_text())
            for t in set(tokens):
                p._tokens[t] = tokens.count(t)
        for t in p._tokens:
            df[t] = df.get(t, 0) + 1

    idf = {t: math.log((N + 1) / (df.get(t, 0) + 1)) for t in set(query_tokens)}

    scores = []
    for p in corpus:
        score = 0.0
        for t in query_tokens:
            tf = p._tokens.get(t, 0)
            if tf:
                score += (1 + math.log(tf)) * idf.get(t, 0)
        scores.append(score)
    return scores


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load_gita(path: Path, enriched_map: dict[str, dict] | None = None) -> list[Passage]:
    passages = []
    if enriched_map is None:
        enriched_map = {}
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        translation = (r.get("translation") or "").strip()
        purport = (r.get("purport") or "").strip()
        if len(translation) < 30:
            continue
        ch = r.get("chapter", "?")
        v = r.get("text_number", "?")
        verse_id = f"BG-{ch}.{v}"
        ref = f"Bhagavad Gita Chapter {ch}, Verse {v}"
        text = translation
        if purport and len(purport) > 40:
            text += " " + purport[:300]
        enriched = enriched_map.get(verse_id, {})
        passages.append(Passage(
            source="Bhagavad Gita",
            ref=ref,
            text=text,
            sanskrit=enriched.get("sanskrit", ""),
            transliteration=enriched.get("transliteration", ""),
        ))
    return passages


def _load_hitopadesha(path: Path) -> list[Passage]:
    passages = []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        title = r.get("story_title", "")
        # Pull all verse translations
        verses = r.get("verses") or []
        verse_texts = [v.get("translation", "") for v in verses if v.get("translation")]
        full = " ".join(verse_texts).strip()
        if len(full) < 80:
            continue
        moral = r.get("moral", "")
        text = full[:500]
        if moral and len(moral) > 20:
            text += f" Moral: {moral}"
        passages.append(Passage(
            source="Hitopadesha",
            ref=f"Story: {title}",
            text=text,
        ))
    # Also absorb paragraphs from the extracted text
    txt_path = path.parent / "hitopadesha_extracted.txt"
    if txt_path.exists():
        with open(txt_path, encoding="utf-8") as f:
            raw = f.read()
        paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 150]
        seen: set[str] = set()
        for i, para in enumerate(paras[:80]):
            clean = re.sub(r"---\s*Page\s*\d+\s*---", "", para).strip()
            clean = re.sub(r"\s+", " ", clean)
            # Skip web-scraping artefacts (URL headers, page navigation lines)
            clean = re.sub(r"^\d{1,2}/\d{1,2}/\d{2,4}\s+\S+\s+\d+/\d+\s*", "", clean).strip()
            clean = re.sub(r"^www\.\S+\s*", "", clean).strip()
            if (len(clean) > 100
                    and not clean.startswith("http")
                    and "columbia.edu" not in clean
                    and "4/25/12" not in clean
                    and clean not in seen):
                seen.add(clean)
                passages.append(Passage(
                    source="Hitopadesha",
                    ref=f"Hitopadesha passage {i+1}",
                    text=clean[:500],
                ))
    return passages


def _load_vidura(path: Path, enriched_map: dict[str, dict] | None = None) -> list[Passage]:
    passages = []
    if enriched_map is None:
        enriched_map = {}
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        translation = (r.get("translation") or "").strip()
        context = (r.get("context") or "").strip()
        if len(translation) < 80:
            continue
        # Skip header/metadata records
        if "www." in translation or "Sanskrit" in translation[:60]:
            continue
        verse_id = r.get("verse_id", "VN-?")
        text = translation[:500]
        if context and context not in text:
            text += f" ({context})"
        enriched = enriched_map.get(verse_id, {})
        passages.append(Passage(
            source="Vidura Niti",
            ref=f"Vidura Niti {verse_id}",
            text=text,
            sanskrit=enriched.get("sanskrit", "") or r.get("sanskrit_shloka", ""),
            transliteration=enriched.get("transliteration", ""),
        ))
    # Supplement with extracted text
    txt_path = path.parent / "vidura_niti_extracted.txt"
    if txt_path.exists():
        with open(txt_path, encoding="utf-8") as f:
            raw = f.read()
        paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 150]
        seen = set()
        for i, para in enumerate(paras[:120]):
            clean = re.sub(r"---\s*Page\s*\d+\s*---", "", para).strip()
            clean = re.sub(r"\s+", " ", clean)
            if len(clean) > 100 and not clean.startswith("http") and clean not in seen:
                seen.add(clean)
                passages.append(Passage(
                    source="Vidura Niti",
                    ref=f"Vidura Niti passage {i+1}",
                    text=clean[:500],
                ))
    return passages


def _load_chanakya(path: Path) -> list[Passage]:
    passages = []
    # Primary: extracted text paragraphs (JSON has no quote content)
    txt_path = path.parent / "chanakya_extracted.txt"
    if txt_path.exists():
        with open(txt_path, encoding="utf-8") as f:
            raw = f.read()
        # Split on page markers and gather rich paragraphs
        paras = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 150]
        seen = set()
        for i, para in enumerate(paras[:200]):
            clean = re.sub(r"---\s*Page\s*\d+\s*---", "", para).strip()
            clean = re.sub(r"\s+", " ", clean)
            if (len(clean) > 120
                    and not clean.startswith("http")
                    and "Rupa Publications" not in clean
                    and "ISBN" not in clean
                    and clean not in seen):
                seen.add(clean)
                passages.append(Passage(
                    source="Chanakya Niti",
                    ref=f"Chanakya Niti passage {i+1}",
                    text=clean[:500],
                ))
    # Also absorb chapter titles from JSON
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        title = (r.get("chapter_title") or "").strip()
        commentary = (r.get("author_commentary") or "").strip()
        if len(commentary) > 150 and "Radhakrishnan" not in commentary[:60]:
            ref = f"Chanakya in Daily Life — {title}" if title and title != "---" else "Chanakya Niti"
            passages.append(Passage(
                source="Chanakya Niti",
                ref=ref,
                text=commentary[:500],
            ))
    return passages


# ── Enriched-shloka loader (proper Unicode Devanagari for all 4 sources) ────────

def _load_enriched_map() -> dict[str, dict]:
    """Build a {verse_id: shloka_dict} map from enriched_sholkas.json."""
    path = DATA_DIR / "enriched_sholkas.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result: dict[str, dict] = {}
    for _src, src_data in data.get("sources", {}).items():
        for shloka in src_data.get("shlokas", []):
            vid = shloka.get("verse_id")
            if vid:
                result[vid] = shloka
    return result


def _load_enriched_passages(enriched_map: dict[str, dict]) -> list[Passage]:
    """Create Passages directly from enriched shlokas (always have proper Devanagari)."""
    passages = []
    for verse_id, shloka in enriched_map.items():
        if verse_id.startswith("BG-"):
            source = "Bhagavad Gita"
            ch = shloka.get("chapter", "?")
            v = shloka.get("verse_number", "?")
            ref = f"Bhagavad Gita Chapter {ch}, Verse {v}"
        elif verse_id.startswith("H-"):
            source = "Hitopadesha"
            ref = f"Hitopadesha {verse_id}"
        elif verse_id.startswith("VN-"):
            source = "Vidura Niti"
            ref = f"Vidura Niti {verse_id}"
        elif verse_id.startswith("CN-"):
            source = "Chanakya Niti"
            ref = f"Chanakya Niti {verse_id}"
        else:
            continue
        translation = (shloka.get("translation") or "").strip()
        interpretation = (shloka.get("interpretation") or "").strip()
        text = translation
        if interpretation and len(interpretation) > 30:
            text += " " + interpretation[:300]
        passages.append(Passage(
            source=source,
            ref=ref,
            text=text[:500],
            sanskrit=shloka.get("sanskrit", ""),
            transliteration=shloka.get("transliteration", ""),
        ))
    return passages


# ── Corpus singleton ───────────────────────────────────────────────────────────

_corpus: list[Passage] = []
_enriched_map: dict[str, dict] = {}


def _build_corpus():
    global _corpus, _enriched_map
    if _corpus:
        return
    _enriched_map = _load_enriched_map()
    # High-quality passages with proper Devanagari go in first
    _corpus += _load_enriched_passages(_enriched_map)
    # Broader corpora for wider retrieval coverage
    _corpus += _load_gita(DATA_DIR / "bhagavad_gita_complete.json", _enriched_map)
    _corpus += _load_hitopadesha(DATA_DIR / "hitopadesha.json")
    _corpus += _load_vidura(DATA_DIR / "vidura_niti.json", _enriched_map)
    _corpus += _load_chanakya(DATA_DIR / "chanakya_in_daily_life.json")


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int = 6) -> list[Passage]:
    """Return up to `top_k` passages most relevant to the query."""
    _build_corpus()
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = _tfidf_scores(tokens, _corpus)

    # Apply 2x boost for passages with proper Devanagari Sanskrit so they
    # consistently outrank raw passages that have no verse text.
    boosted = [
        (score * (2.0 if _has_devanagari(p.sanskrit) else 1.0), p)
        for score, p in zip(scores, _corpus)
    ]
    ranked = sorted(boosted, key=lambda x: x[0], reverse=True)

    # Ensure we draw from at least 2 different sources when possible
    seen_sources: dict[str, int] = {}
    results: list[Passage] = []
    for score, passage in ranked:
        if score <= 0:
            break
        src_count = seen_sources.get(passage.source, 0)
        if src_count >= 2:
            continue  # cap per-source at 2 to keep diversity
        seen_sources[passage.source] = src_count + 1
        results.append(passage)
        if len(results) >= top_k:
            break

    # Guarantee at least one Sanskrit-bearing passage is present.
    # If none made it in, swap in the highest-scoring Sanskrit passage.
    if results and not any(_has_devanagari(p.sanskrit) for p in results):
        for score, passage in ranked:
            if _has_devanagari(passage.sanskrit):
                results[-1] = passage  # replace lowest-ranked slot
                break

    return results


def _has_devanagari(text: str) -> bool:
    """Return True if text contains at least one Unicode Devanagari character."""
    return any('\u0900' <= c <= '\u097F' for c in text)


def format_passages_for_prompt(passages: list[Passage]) -> str:
    """Format retrieved passages as structured context for the LLM prompt."""
    if not passages:
        return ""
    lines = ["=== RETRIEVED KNOWLEDGE BASE PASSAGES ===\n"]
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] SOURCE: {p.source} | REF: {p.ref}")
        if p.sanskrit and _has_devanagari(p.sanskrit):
            lines.append(f"    Sanskrit: {p.sanskrit[:200]}")
        if p.transliteration and len(p.transliteration) > 10 and not _has_devanagari(p.transliteration):
            lines.append(f"    Transliteration: {p.transliteration[:200]}")
        lines.append(f"    Teaching: {p.text[:250]}")
        lines.append("")
    lines.append("=== END OF KNOWLEDGE BASE ===")
    return "\n".join(lines)


# Quick self-test
if __name__ == "__main__":
    results = retrieve("I feel depressed and lost in life", top_k=6)
    print(f"Got {len(results)} passages:")
    for p in results:
        print(f"  [{p.source}] {p.ref}: {p.text[:80]}…")
