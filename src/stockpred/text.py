"""10-K text analysis: section extraction, dictionary sentiment and year-over-year change.

Pipeline per company:
  filing list -> download 10-K HTML -> plain text -> Item 1A (Risk Factors) and
  Item 7 (MD&A) -> tokens -> Loughran-McDonald sentiment + similarity to last year.

Why Loughran-McDonald and not a generic sentiment lexicon? Generic lists mark words like
"liability", "tax" or "cost" as negative, but in a 10-K they are neutral accounting
terms. LM was built specifically from 10-K filings (Loughran & McDonald, J. of Finance 2011).
"""
from __future__ import annotations

import gzip
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from .fundamentals import fiscal_year_of

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

SECTIONS = ("risk_factors", "mdna")

LM_CATEGORIES = ("Negative", "Positive", "Uncertainty", "Litigious", "Constraining")


# --------------------------------------------------------------------------- filings
def list_10k_filings(client, cik: int, start_year: int = 2010) -> pd.DataFrame:
    """Original 10-K filings (no amendments) with their fiscal year and document URL."""
    df = pd.DataFrame(client.all_filings(cik, since_year=start_year))
    df = df[df["form"] == "10-K"].copy()
    if df.empty:
        return df
    df["report_date"] = pd.to_datetime(df["reportDate"], errors="coerce")
    df["filing_date"] = pd.to_datetime(df["filingDate"])
    df = df.dropna(subset=["report_date"])
    df["fiscal_year"] = fiscal_year_of(df["report_date"])
    df = df[df["fiscal_year"] >= start_year]
    acc = df["accessionNumber"].str.replace("-", "", regex=False)
    df["url"] = [f"{ARCHIVE}/{cik}/{a}/{doc}" for a, doc in zip(acc, df["primaryDocument"])]
    df["cik"] = cik
    df = df.sort_values("filing_date").drop_duplicates("fiscal_year", keep="first")
    return df[["fiscal_year", "report_date", "filing_date", "cik", "accessionNumber", "url"]] \
        .rename(columns={"accessionNumber": "accession"}).reset_index(drop=True)


# ----------------------------------------------------------------- HTML -> sections
BLOCK_TAGS = ["p", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"]


def html_to_text(html: str) -> str:
    """Plain text with line breaks ONLY after block elements.

    A naive soup.get_text("\n") also breaks lines around inline <a>/<span> tags, so a
    sentence like 'see <a>Item 1A</a> in this report' turns 'Item 1A' into its own
    line - indistinguishable from a real section heading."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for cell in soup.find_all(["td", "th"]):
        cell.append(" ")                       # keep table cells apart
    for block in soup.find_all(BLOCK_TAGS):
        block.append("\n")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text)


def _item_regex(item: str) -> re.Pattern:
    # a heading line: "Item 1A." / "ITEM 1A: RISK FACTORS" - short, at line start
    return re.compile(rf"(?im)^\s*item\s*{item}\s*[\.:\-–—]?(?![0-9a-z])[^\n]{{0,150}}$")


# fallback for 10-Ks organised by a cross-reference index (banks, ExxonMobil...)
MDNA_HEADING = re.compile(r"(?im)^\s*management[’'`]?s\s+discussion\s+and\s+analysis\b[^\n]{0,150}$")
MDNA_END = re.compile(
    r"(?im)^\s*(quantitative\s+and\s+qualitative\s+disclosures|financial\s+statements\s+and\s+"
    r"supplementary|report\s+of\s+independent\s+registered|management[’'`]?s\s+report\s+on\s+"
    r"internal\s+control)[^\n]{0,150}$"
)


def _longest_span(text: str, starts: list[int], ends: list[int], min_chars: int) -> str | None:
    """Pair every start with the first end after it; keep the longest span.
    Table-of-contents entries produce tiny spans, the real section a long one."""
    ends = sorted(ends)
    best = ""
    for s in starts:
        nxt = next((e for e in ends if e > s), None)
        if nxt is not None and nxt - s > len(best):
            best = text[s:nxt]
    return best.strip() if len(best) >= min_chars else None


def extract_section(text: str, start_item: str, end_items: tuple[str, ...],
                    min_chars: int = 2_000) -> str | None:
    """Text between the 'Item <start>' heading and the next 'Item <end>' heading."""
    starts = [m.start() for m in _item_regex(start_item).finditer(text)]
    ends = [m.start() for e in end_items for m in _item_regex(e).finditer(text)]
    return _longest_span(text, starts, ends, min_chars)


def extract_mdna(text: str, min_chars: int = 2_000) -> str | None:
    """MD&A via 'Item 7', falling back to its title when the 10-K uses a cross-reference index."""
    section = extract_section(text, "7", ("7A", "8"), min_chars)
    if section and len(section) >= 10_000:
        return section
    starts = [m.start() for m in MDNA_HEADING.finditer(text)]
    ends = [m.start() for m in MDNA_END.finditer(text)]
    fallback = _longest_span(text, starts, ends, min_chars)
    return max([x for x in (section, fallback) if x], key=len, default=None)


EXTRACTORS = {
    "risk_factors": lambda t: extract_section(t, "1A", ("1B", "1C", "2")),
    "mdna": extract_mdna,
}


def get_sections(client, cik: int, filing: pd.Series, cache_dir: Path) -> dict[str, str | None]:
    """Section texts for one filing, cached as gzip text (raw HTML is not kept)."""
    out, missing = {}, []
    for name in SECTIONS:
        path = Path(cache_dir) / f"{cik}" / f"{filing['accession']}_{name}.txt.gz"
        if path.exists():
            content = gzip.decompress(path.read_bytes()).decode("utf-8")
            out[name] = content or None          # empty file = known extraction failure
        else:
            missing.append(name)
    if not missing:
        return out

    text = html_to_text(client.get_text(filing["url"]))
    for name in missing:
        section = EXTRACTORS[name](text)
        path = Path(cache_dir) / f"{cik}" / f"{filing['accession']}_{name}.txt.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(gzip.compress((section or "").encode("utf-8")))
        out[name] = section
    return out


# ------------------------------------------------------------------- NLP features
@lru_cache(maxsize=1)
def lm_dictionary() -> dict[str, frozenset[str]]:
    """Loughran-McDonald word lists, shipped inside the pysentiment2 package."""
    path = resources.files("pysentiment2") / "static" / "LM.csv"
    lm = pd.read_csv(path, usecols=["Word", *LM_CATEGORIES])
    return {c.lower(): frozenset(lm.loc[lm[c] > 0, "Word"].str.upper()) for c in LM_CATEGORIES}


@lru_cache(maxsize=1)
def stopwords() -> frozenset[str]:
    """LM 'Generic' stop-word list (THE, OF, AND...). Without removing these, two
    10-Ks always look ~99 % similar because function words dominate the counts."""
    path = resources.files("pysentiment2") / "static" / "Generic.txt"
    return frozenset(path.read_text(encoding="latin-1").upper().split())


def content_words(tokens: list[str]) -> list[str]:
    stop = stopwords()
    return [t for t in tokens if t not in stop and len(t) > 2]


def tokenize(text: str) -> list[str]:
    """Upper-case alphabetic words (LM dictionary is upper-case); numbers dropped."""
    return re.findall(r"[A-Z][A-Z'\-]*[A-Z]|[A-Z]", text.upper())


def sentiment_scores(tokens: list[str]) -> dict[str, float]:
    """Share of words in each LM category + net tone in [-1, 1]."""
    n = len(tokens)
    counts = Counter(tokens)
    scores = {"n_words": n}
    for cat, words in lm_dictionary().items():
        hits = sum(cnt for w, cnt in counts.items() if w in words)
        scores[f"lm_{cat}"] = hits / n if n else np.nan
    pos, neg = scores["lm_positive"], scores["lm_negative"]
    scores["net_tone"] = (pos - neg) / (pos + neg) if (pos + neg) else np.nan
    return scores


def cosine_similarity(a: list[str], b: list[str]) -> float:
    """Cosine of word-count vectors: 1 = same vocabulary mix, 0 = nothing in common.
    Used by 'Lazy Prices' (Cohen, Malloy, Nguyen 2020) to measure how much a 10-K changed.
    Pass content words (stop words removed) for a meaningful spread."""
    ca, cb = Counter(a), Counter(b)
    dot = sum(ca[w] * cb[w] for w in ca.keys() & cb.keys())
    norm = np.sqrt(sum(v * v for v in ca.values())) * np.sqrt(sum(v * v for v in cb.values()))
    return dot / norm if norm else np.nan


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    """Overlap of the vocabularies (sets of distinct words)."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else np.nan


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n", text)
    return [p.strip() for p in parts if len(p.split()) >= 8]   # skip headings / fragments


def _sentence_words(sentence: str) -> frozenset[str]:
    # numbers and years change every year ("2025" -> "2026") - tokenize() drops them
    return frozenset(content_words(tokenize(sentence)))


def new_sentences(current: str, previous: str, threshold: float = 0.8) -> list[str]:
    """Sentences added this year = newly disclosed risks / topics.

    Exact matching is too strict: companies split, merge or lightly reword sentences.
    A sentence counts as OLD if >= `threshold` of its content words appear together
    in a single sentence of last year's text (containment, not equality, so a
    sentence split out of a longer one is still recognised)."""
    prev_sets = [_sentence_words(s) for s in _sentences(previous)]
    # inverted index word -> previous sentences containing it, to avoid n*m comparisons
    index: dict[str, list[int]] = {}
    for i, words in enumerate(prev_sets):
        for w in words:
            index.setdefault(w, []).append(i)

    added = []
    for sentence in _sentences(current):
        words = _sentence_words(sentence)
        if not words:
            continue
        overlap = Counter(i for w in words for i in index.get(w, ()))
        best = max(overlap.values(), default=0) / len(words)
        if best < threshold:
            added.append(sentence)
    return added


def share_new_sentences(current: str, previous: str) -> float:
    total = sum(1 for s in _sentences(current) if _sentence_words(s))
    return len(new_sentences(current, previous)) / total if total else np.nan


# ----------------------------------------------------------------- whole universe
@dataclass
class TextResult:
    features: pd.DataFrame        # one row per (ticker, fiscal_year, section)
    filings: pd.DataFrame         # which filings were processed


def build_text_features(client, universe: pd.DataFrame, cache_dir: Path,
                        start_year: int = 2010, progress: bool = True,
                        predecessors: dict[int, list[int]] | None = None) -> TextResult:
    from tqdm.auto import tqdm

    rows, filing_rows = [], []
    iterator = tqdm(list(universe.itertuples(index=False)), disable=not progress, desc="10-K text")
    for comp in iterator:
        ciks = [int(comp.cik), *(predecessors or {}).get(int(comp.cik), [])]
        filings = pd.concat([list_10k_filings(client, c, start_year) for c in ciks], ignore_index=True)
        if filings.empty:
            print(f"  ! {comp.ticker}: no 10-K filings found")
            continue
        filings = filings.sort_values("filing_date").drop_duplicates("fiscal_year").reset_index(drop=True)
        prev_tokens: dict[str, list[str] | None] = {}
        prev_text: dict[str, str | None] = {}
        row_prev_len: dict[str, int] = {}
        for _, f in filings.iterrows():
            iterator.set_postfix_str(f"{comp.ticker} FY{f['fiscal_year']}")
            try:
                sections = get_sections(client, int(f["cik"]), f, cache_dir)
            except Exception as exc:                     # network error, odd document...
                sections = {name: None for name in SECTIONS}
                print(f"  ! {comp.ticker} FY{f['fiscal_year']}: {exc}")
            filing_rows.append({"ticker": comp.ticker, **f.to_dict()})

            for name, txt in sections.items():
                row = {"ticker": comp.ticker, "fiscal_year": int(f["fiscal_year"]),
                       "section": name, "filing_date": f["filing_date"],
                       "extracted": txt is not None}
                tokens = tokenize(txt) if txt else None
                if tokens:
                    row.update(sentiment_scores(tokens))
                    content = content_words(tokens)
                    prev = prev_tokens.get(name)
                    if prev:
                        row["sim_cosine"] = cosine_similarity(content, prev)
                        row["sim_jaccard"] = jaccard_similarity(content, prev)
                        row["length_change"] = len(tokens) / row_prev_len[name] - 1
                        row["share_new_sentences"] = share_new_sentences(txt, prev_text[name])
                    row_prev_len[name] = len(tokens)
                    prev_tokens[name] = content
                else:
                    prev_tokens[name] = None     # a failed year breaks the chain on purpose
                prev_text[name] = txt
                rows.append(row)

    return TextResult(pd.DataFrame(rows), pd.DataFrame(filing_rows))


def load_section(cache_dir: Path, cik: int, accession: str, section: str) -> str | None:
    path = Path(cache_dir) / f"{cik}" / f"{accession}_{section}.txt.gz"
    if not path.exists():
        return None
    return gzip.decompress(path.read_bytes()).decode("utf-8") or None
