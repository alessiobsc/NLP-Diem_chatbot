"""
Benchmark: current pipeline (trafilatura) vs Crawl4AI (filtered).

Both extractors are configured to mirror the same filtering behaviour as closely as possible.

FILTER EQUIVALENCES
───────────────────
Trafilatura                          Crawl4AI equivalent          Fidelity
────────────────────────────────     ─────────────────────────    ─────────
include_links=False                  ignore_links=True             exact
include_images=False                 ignore_images=True            exact
include_tables=True                  no tables exclusion           exact
BS4 fallback: remove                 excluded_tags=[nav,footer,    exact
  nav/footer/header/aside/           header,aside,iframe,
  iframe/noscript/script/style       noscript,script,style]
favor_precision=True                 PruningContentFilter          APPROX.
  (trafilatura ML classifier:         (threshold=0.48, fixed)      Trafilatura uses its own
   scores blocks by content/link      Crawl4AI scores by text      content classifier;
   density and boilerplate patterns)  density and link density     PruningContentFilter uses
                                                                   a similar heuristic but
                                                                   is NOT identical. This is
                                                                   the only unavoidable gap.

CONSEQUENCE: on pages where trafilatura's classifier decides to discard the entire main
content (e.g. JS-heavy homepages), trafilatura falls back to _bs4_extractor (no
precision filter). Crawl4AI in those same cases will keep more content because
PruningContentFilter is less aggressive than the trafilatura classifier.
This asymmetry is reported honestly in the results.

Level A — Same HTML source, different extraction (isolates extractor quality)
Level B — Playwright fetch vs requests (shows JS-rendering benefit)

Run with:
    python scripts/compare_crawlers.py

Prerequisites:
    crawl4ai-setup    (or: playwright install chromium)
"""

import asyncio
import traceback
import difflib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import urllib3
import requests as req_lib

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.enrichment import HEADER_KEYWORDS, is_meaningful_line
from src.ingestion.parser import extract_html_metadata, html_extractor

# ---------------------------------------------------------------------------
# Test URLs
# ---------------------------------------------------------------------------

TEST_URLS = [
    {"url": "https://www.diem.unisa.it/", "category": "homepage"},
    {"url": "https://www.diem.unisa.it/dipartimento/presentazione", "category": "presentation"},
    {"url": "https://www.diem.unisa.it/didattica/offerta-formativa", "category": "offerta_formativa"},
    {"url": "https://www.diem.unisa.it/dipartimento/eccellenza/strutture", "category": "structures"},
    {"url": "https://www.diem.unisa.it/ricerca/premi-ricerca", "category": "research_prizes"},
    {"url": "https://docenti.unisa.it/004322/home", "category": "docente_home"},
    {"url": "https://docenti.unisa.it/004322/didattica", "category": "docente_didattica"},
    {"url": "https://corsi.unisa.it/ingegneria-informatica", "category": "course_home"},
    {"url": "https://corsi.unisa.it/ingegneria-informatica/didattica/insegnamenti", "category": "course_insegnamenti"},
    {"url": "https://corsi.unisa.it/ingegneria-informatica/immatricolazioni", "category": "immatricolazioni"},
]

RATE_LIMIT_S = 1.5
REQUEST_TIMEOUT = 15
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "crawler_comparison"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    text_length: int = 0
    line_count: int = 0
    noise_ratio: float = 0.0
    keyword_density: float = 0.0
    structural_score: int = 0

    def delta(self, other: "Metrics") -> dict:
        """Return per-field delta (self - other), as dict."""
        return {
            "text_length": self.text_length - other.text_length,
            "line_count": self.line_count - other.line_count,
            "noise_ratio": round(self.noise_ratio - other.noise_ratio, 4),
            "keyword_density": round(self.keyword_density - other.keyword_density, 4),
            "structural_score": self.structural_score - other.structural_score,
        }


@dataclass
class FactPreservation:
    """Comparison of atomic facts between two extractions (trafilatura vs Crawl4AI)."""
    facts_in_traf: int = 0       # unique facts in trafilatura
    facts_in_c4ai: int = 0       # unique facts in crawl4ai
    facts_shared: int = 0        # in both
    only_in_traf: int = 0        # present in traf but lost by c4ai
    only_in_c4ai: int = 0        # new in c4ai, not in traf
    recall_by_c4ai: float | None = None  # shared / facts_in_traf (how much of traf does c4ai preserve); None if facts_in_traf=0
    recall_by_traf: float | None = None  # shared / facts_in_c4ai (how much of c4ai is also in traf); None if facts_in_c4ai=0


@dataclass
class ExtractionResult:
    method: str
    text: str = ""
    metrics: Metrics = field(default_factory=Metrics)
    fetch_time_s: float = 0.0
    extract_time_s: float = 0.0
    error: str | None = None


@dataclass
class URLComparison:
    url: str
    category: str
    html_metadata: dict = field(default_factory=dict)
    level_a_trafilatura: ExtractionResult = field(default_factory=lambda: ExtractionResult("trafilatura"))
    level_a_crawl4ai: ExtractionResult = field(default_factory=lambda: ExtractionResult("crawl4ai"))
    level_b_trafilatura: ExtractionResult = field(default_factory=lambda: ExtractionResult("trafilatura"))
    level_b_crawl4ai: ExtractionResult = field(default_factory=lambda: ExtractionResult("crawl4ai_playwright"))
    content_diff_ratio: float | None = None
    fact_preservation_a: FactPreservation = field(default_factory=FactPreservation)
    fact_preservation_b: FactPreservation = field(default_factory=FactPreservation)

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_dependencies() -> bool:
    ok = True
    try:
        from crawl4ai import AsyncWebCrawler  # noqa: F401
    except ImportError:
        print("ERROR: crawl4ai not installed. Run: pip install crawl4ai")
        ok = False
    try:
        import trafilatura  # noqa: F401
    except ImportError:
        print("ERROR: trafilatura not installed.")
        ok = False
    return ok


async def check_playwright() -> bool:
    """Warn if Playwright browsers are not installed (non-fatal, Level B will fail gracefully)."""
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            await browser.close()
        return True
    except Exception:
        traceback.print_exc()
        print(
            "WARNING: Playwright browsers not found or not launchable — Level B (Playwright fetch) will be skipped.\n"
            "         To fix: run  crawl4ai-setup  or  playwright install chromium\n"
        )
        return False

# ---------------------------------------------------------------------------
# Fact extraction & preservation
# ---------------------------------------------------------------------------

# Regex patterns for atomic facts worth preserving
_RE_NUMBERS = re.compile(r"\b\d+(?:[.,]\d+)?\b")                     # 18, 3.5, 2024, 36,5
_RE_EMAILS = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.I)
_RE_ACRONYMS = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:[/\-][A-Z0-9]+)*\b")  # DIEM, TOLC, ING-INF/05, LM32
_RE_SSD = re.compile(r"\b[A-Z]{2,}-[A-Z0-9]{2,}/[A-Z0-9]+\b")       # ING-INF/05, MAT/05


def extract_facts(text: str) -> set[str]:
    """Extract atomic facts from text: numbers, emails, acronyms, SSD codes."""
    facts: set[str] = set()
    facts.update(_RE_NUMBERS.findall(text))
    facts.update(e.lower() for e in _RE_EMAILS.findall(text))
    facts.update(_RE_ACRONYMS.findall(text))
    facts.update(_RE_SSD.findall(text))
    # Normalise: strip trivial single-char or pure-zero entries
    return {f for f in facts if len(f) > 1 and f not in ("00", "0")}


def compute_fact_preservation(text_traf: str, text_c4ai: str) -> FactPreservation:
    """Compare fact sets between trafilatura and Crawl4AI extractions."""
    if not text_traf and not text_c4ai:
        return FactPreservation()
    facts_t = extract_facts(text_traf)
    facts_c = extract_facts(text_c4ai)
    shared = facts_t & facts_c
    recall_by_c4ai = round(len(shared) / len(facts_t), 4) if facts_t else None
    recall_by_traf = round(len(shared) / len(facts_c), 4) if facts_c else None
    return FactPreservation(
        facts_in_traf=len(facts_t),
        facts_in_c4ai=len(facts_c),
        facts_shared=len(shared),
        only_in_traf=len(facts_t - facts_c),
        only_in_c4ai=len(facts_c - facts_t),
        recall_by_c4ai=recall_by_c4ai,
        recall_by_traf=recall_by_traf,
    )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(text: str, is_markdown: bool = False) -> Metrics:
    if not text:
        return Metrics()

    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    line_count = len(non_empty)

    # noise_ratio: proportion of lines that are NOT meaningful
    if line_count:
        noise_count = sum(1 for l in non_empty if not is_meaningful_line(l))
        noise_ratio = noise_count / line_count
    else:
        noise_ratio = 0.0

    # keyword_density: total keyword hits / total words
    words = text.lower().split()
    total_words = len(words) or 1
    text_lower = text.lower()
    keyword_hits = sum(text_lower.count(kw) for kw in HEADER_KEYWORDS)
    keyword_density = keyword_hits / total_words

    # structural_score
    if is_markdown:
        structural_score = sum(1 for l in lines if re.match(r"^#{1,6}\s", l))
    else:
        # trafilatura: count short all-caps-like lines (section headers in plain text)
        structural_score = sum(
            1 for l in non_empty
            if 5 <= len(l.strip()) <= 60 and l.strip() == l.strip().upper() and l.strip().isalpha()
        )

    return Metrics(
        text_length=len(text),
        line_count=line_count,
        noise_ratio=round(noise_ratio, 4),
        keyword_density=round(keyword_density, 6),
        structural_score=structural_score,
    )

# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_html(url: str) -> tuple[str, float]:
    """Fetch raw HTML with requests. Returns (html, elapsed_s)."""
    t0 = time.perf_counter()
    r = req_lib.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    return r.text, round(time.perf_counter() - t0, 3)

# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_trafilatura(html: str, fetch_time: float = 0.0) -> ExtractionResult:
    t0 = time.perf_counter()
    try:
        text = html_extractor(html)
        elapsed = round(time.perf_counter() - t0, 3)
        return ExtractionResult(
            method="trafilatura",
            text=text,
            metrics=compute_metrics(text, is_markdown=False),
            fetch_time_s=fetch_time,
            extract_time_s=elapsed,
        )
    except Exception as e:
        return ExtractionResult(method="trafilatura", error=str(e), fetch_time_s=fetch_time)


async def extract_crawl4ai_local(crawler, html: str, url: str, config) -> ExtractionResult:
    """Level A: pass already-fetched HTML through Crawl4AI filtered markdown pipeline.

    Uses fit_markdown (output of PruningContentFilter) as primary text, analogous to
    trafilatura's precision-filtered output. Falls back to raw_markdown (post
    excluded_tags removal) if PruningContentFilter discards everything — analogous
    to trafilatura's own BS4 fallback.
    """
    t0 = time.perf_counter()
    try:
        result = await crawler.aprocess_html(
            url=url,
            html=html,
            extracted_content="",
            config=config,
            screenshot_data="",
            pdf_data="",
            verbose=False,
        )
        elapsed = round(time.perf_counter() - t0, 3)
        md = result.markdown if result else None
        # fit_markdown = PruningContentFilter output (analog to favor_precision=True)
        # raw_markdown = after excluded_tags removal only (analog to BS4 fallback)
        text = (md.fit_markdown or md.raw_markdown) if md else ""
        return ExtractionResult(
            method="crawl4ai",
            text=text,
            metrics=compute_metrics(text, is_markdown=True),
            fetch_time_s=0.0,
            extract_time_s=elapsed,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return ExtractionResult(method="crawl4ai", error=f"{type(e).__name__}: {e}\n{tb}")


async def extract_crawl4ai_full(crawler, url: str, config) -> ExtractionResult:
    """Level B: Playwright fetch + filtered markdown extraction.

    Same filter chain as Level A (excluded_tags + PruningContentFilter).
    Uses fit_markdown with fallback to raw_markdown, same as extract_crawl4ai_local.
    """
    t0 = time.perf_counter()
    try:
        result = await crawler.arun(url=url, config=config)
        elapsed = round(time.perf_counter() - t0, 3)
        if not result or not result.success:
            err = getattr(result, "error_message", "unknown error") if result else "no result"
            return ExtractionResult(method="crawl4ai_full", error=err)
        md = result.markdown
        text = (md.fit_markdown or md.raw_markdown) if md else ""
        return ExtractionResult(
            method="crawl4ai_full",
            text=text,
            metrics=compute_metrics(text, is_markdown=True),
            fetch_time_s=0.0,
            extract_time_s=elapsed,
        )
    except Exception as e:
        tb = traceback.format_exc()
        return ExtractionResult(method="crawl4ai_full", error=f"{type(e).__name__}: {e}\n{tb}")

# ---------------------------------------------------------------------------
# Per-URL orchestration
# ---------------------------------------------------------------------------

async def compare_url(entry: dict, crawler, cfg_local, cfg_full, playwright_ok: bool) -> URLComparison:
    url = entry["url"]
    category = entry["category"]
    comp = URLComparison(url=url, category=category)

    # Fetch HTML with requests (shared for Level A + Level B trafilatura side)
    html = ""
    fetch_time = 0.0
    fetch_error: str | None = None
    try:
        html, fetch_time = fetch_html(url)
        comp.html_metadata = extract_html_metadata(html)
    except Exception as e:
        fetch_error = f"fetch failed: {e}"

    # Level A — same HTML, different extraction (both skip if fetch failed)
    if fetch_error:
        comp.level_a_trafilatura.error = fetch_error
        comp.level_a_crawl4ai.error = fetch_error
        comp.level_b_trafilatura.error = fetch_error
    else:
        comp.level_a_trafilatura = extract_trafilatura(html, fetch_time=fetch_time)
        comp.level_a_crawl4ai = await extract_crawl4ai_local(crawler, html, url, cfg_local)
        comp.fact_preservation_a = compute_fact_preservation(
            comp.level_a_trafilatura.text, comp.level_a_crawl4ai.text
        )
        comp.level_b_trafilatura = extract_trafilatura(html, fetch_time=fetch_time)

    # Level B Crawl4AI — Playwright fetches independently (may succeed even if requests failed)
    if playwright_ok:
        comp.level_b_crawl4ai = await extract_crawl4ai_full(crawler, url, cfg_full)
        # content_diff_ratio: compare Level A c4ai (requests-fetched HTML) vs Level B c4ai (Playwright-fetched HTML)
        # with the SAME extractor on both sides — isolates the JS rendering benefit of Playwright.
        # High ratio (→1.0) = Playwright adds no JS content. Low ratio = Playwright fetched extra JS content.
        c4ai_a_text = comp.level_a_crawl4ai.text if not fetch_error else ""
        if c4ai_a_text and comp.level_b_crawl4ai.text:
            comp.content_diff_ratio = round(
                difflib.SequenceMatcher(None, c4ai_a_text, comp.level_b_crawl4ai.text).ratio(), 4
            )
        traf_text = comp.level_b_trafilatura.text if not fetch_error else ""
        comp.fact_preservation_b = compute_fact_preservation(traf_text, comp.level_b_crawl4ai.text)
    else:
        comp.level_b_crawl4ai.error = "skipped (Playwright not installed)"

    return comp

# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

METRIC_LABELS = ["text_length", "line_count", "noise_ratio", "keyword_density", "structural_score"]

def section(title: str) -> None:
    """Print a formatted section header."""
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)

def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)

def _delta_str(d, key) -> str:
    v = d[key]
    if isinstance(v, float):
        return f"{v:+.4f}"
    return f"{v:+d}" if v != 0 else "  0"


def print_summary_table(comparisons: list[URLComparison]) -> None:
    sep = "=" * 90
    thin = "-" * 90

    def print_url_block(url: str, category: str, r_traf: ExtractionResult, r_c4ai: ExtractionResult, diff_ratio=None):
        label = f"{category} | {url[:55]}"
        print(f"\n  {label}")
        print(f"  {thin[:len(label)+2]}")
        header = f"  {'Metric':<20} {'Trafilatura':>14} {'Crawl4AI':>14} {'Delta':>12}"
        print(header)
        print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*12}")

        if r_traf.error and r_c4ai.error:
            print(f"  ERROR traf: {r_traf.error}")
            print(f"  ERROR c4ai: {r_c4ai.error}")
            return

        m_t = r_traf.metrics
        m_c = r_c4ai.metrics
        d = m_c.delta(m_t) if not (r_traf.error or r_c4ai.error) else {}

        for key in METRIC_LABELS:
            vt = _fmt(getattr(m_t, key)) if not r_traf.error else "ERROR"
            vc = _fmt(getattr(m_c, key)) if not r_c4ai.error else "ERROR"
            dv = _delta_str(d, key) if d else "  N/A"
            print(f"  {key:<20} {vt:>14} {vc:>14} {dv:>12}")

        if diff_ratio is not None:
            print(f"  {'c4ai_A→B_diff':<20} {'':>14} {'':>14} {diff_ratio:>12.4f}  (c4ai requests vs c4ai playwright)")

        et = f"{r_traf.extract_time_s:.3f}s" if not r_traf.error else "N/A"
        ec = f"{r_c4ai.extract_time_s:.3f}s" if not r_c4ai.error else "N/A"
        print(f"  {'extract_time':<20} {et:>14} {ec:>14}")

    def print_fact_block(label: str, fp: FactPreservation):
        print(f"\n  Fact Preservation ({label})")
        print(f"  {'facts_in_traf':<22} {fp.facts_in_traf:>6}  |  {'facts_in_c4ai':<22} {fp.facts_in_c4ai:>6}")
        print(f"  {'shared':<22} {fp.facts_shared:>6}  |  {'only_in_traf':<22} {fp.only_in_traf:>6}  |  {'only_in_c4ai':<22} {fp.only_in_c4ai:>6}")
        rc = f"{fp.recall_by_c4ai:.2%}" if fp.recall_by_c4ai is not None else "  N/A"
        rt = f"{fp.recall_by_traf:.2%}" if fp.recall_by_traf is not None else "  N/A"
        print(f"  {'recall_by_c4ai (traf→c4ai)':<30} {rc:>6}  |  {'recall_by_traf (c4ai→traf)':<30} {rt:>6}")

    section("LEVEL A — Same HTML, different extraction (isolates extractor quality)")
    for comp in comparisons:
        print_url_block(comp.url, comp.category, comp.level_a_trafilatura, comp.level_a_crawl4ai)
        print_fact_block("Level A", comp.fact_preservation_a)

    section("LEVEL B — Playwright fetch vs requests fetch (JS rendering benefit)")
    for comp in comparisons:
        print_url_block(
            comp.url, comp.category,
            comp.level_b_trafilatura, comp.level_b_crawl4ai,
            diff_ratio=comp.content_diff_ratio,
        )
        if comp.level_b_crawl4ai.error != "skipped (Playwright not installed)":
            print_fact_block("Level B", comp.fact_preservation_b)

    # Aggregate averages (Level A only, skip errored)
    section("AGGREGATE AVERAGES — Level A")
    valid = [c for c in comparisons if not c.level_a_trafilatura.error and not c.level_a_crawl4ai.error]
    if valid:
        print(f"\n  {'Metric':<20} {'Trafilatura':>14} {'Crawl4AI':>14} {'Delta':>12}")
        print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*12}")
        for key in METRIC_LABELS:
            avg_t = sum(getattr(c.level_a_trafilatura.metrics, key) for c in valid) / len(valid)
            avg_c = sum(getattr(c.level_a_crawl4ai.metrics, key) for c in valid) / len(valid)
            delta = avg_c - avg_t
            fmt_t = _fmt(avg_t) if isinstance(avg_t, float) and key in ("noise_ratio", "keyword_density") else f"{avg_t:.1f}"
            fmt_c = _fmt(avg_c) if isinstance(avg_c, float) and key in ("noise_ratio", "keyword_density") else f"{avg_c:.1f}"
            fmt_d = f"{delta:+.4f}" if key in ("noise_ratio", "keyword_density") else f"{delta:+.1f}"
            print(f"  {key:<20} {fmt_t:>14} {fmt_c:>14} {fmt_d:>12}")
    print()

# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _result_to_dict(r: ExtractionResult) -> dict:
    d = asdict(r)
    d["metrics"] = asdict(r.metrics)
    return d


def save_results_json(comparisons: list[URLComparison]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"results_{ts}.json"

    def comp_to_dict(comp: URLComparison) -> dict:
        return {
            "url": comp.url,
            "category": comp.category,
            "html_metadata": comp.html_metadata,
            "level_a": {
                "trafilatura": _result_to_dict(comp.level_a_trafilatura),
                "crawl4ai": _result_to_dict(comp.level_a_crawl4ai),
                "fact_preservation": asdict(comp.fact_preservation_a),
            },
            "level_b": {
                "trafilatura": _result_to_dict(comp.level_b_trafilatura),
                "crawl4ai": _result_to_dict(comp.level_b_crawl4ai),
                "c4ai_playwright_vs_requests_diff": comp.content_diff_ratio,
                "fact_preservation": asdict(comp.fact_preservation_b),
            },
        }

    # Build aggregate summary
    valid_a = [c for c in comparisons if not c.level_a_trafilatura.error and not c.level_a_crawl4ai.error]
    summary: dict = {"level_a": {}}
    if valid_a:
        for method, attr in [("trafilatura", "level_a_trafilatura"), ("crawl4ai", "level_a_crawl4ai")]:
            summary["level_a"][method] = {
                key: round(sum(getattr(getattr(c, attr).metrics, key) for c in valid_a) / len(valid_a), 4)
                for key in METRIC_LABELS
            }

    output = {
        "timestamp": datetime.now().isoformat(),
        "test_urls_count": len(comparisons),
        "valid_level_a": len(valid_a),
        "summary": summary,
        "comparisons": [comp_to_dict(c) for c in comparisons],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_comparison(run_retrieval: bool = False) -> None:
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    from crawl4ai.content_filter_strategy import PruningContentFilter

    playwright_ok = await check_playwright()
    print(f"Playwright availability check returned: {playwright_ok}")

    # Filtered Crawl4AI config — mirrors trafilatura/BS4 filtering as closely as possible.
    # See module docstring for the full equivalence table and known limitations.
    _md_gen = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed"),
        options={"ignore_links": True, "ignore_images": True},
    )
    _excluded = ["nav", "footer", "header", "aside", "iframe", "noscript", "script", "style"]

    # Level A: same HTML already fetched — no browser config needed
    cfg_local = CrawlerRunConfig(
        excluded_tags=_excluded,
        markdown_generator=_md_gen,
        word_count_threshold=10,
    )
    # Level B: Playwright fetch — add page load params
    cfg_full = CrawlerRunConfig(
        excluded_tags=_excluded,
        markdown_generator=_md_gen,
        word_count_threshold=10,
        wait_until="domcontentloaded",
        page_timeout=15000,
    )

    comparisons: list[URLComparison] = []
    total = len(TEST_URLS)

    print(f"\nCrawler Comparison Benchmark — {total} URLs, 2 levels")
    print("=" * 60)

    async with AsyncWebCrawler() as crawler:
        for i, entry in enumerate(TEST_URLS, 1):
            print(f"[{i:02d}/{total}] {entry['category']} | {entry['url']}")
            comp = await compare_url(entry, crawler, cfg_local, cfg_full, playwright_ok)
            comparisons.append(comp)
            if i < total:
                await asyncio.sleep(RATE_LIMIT_S)

    print_summary_table(comparisons)
    path = save_results_json(comparisons)
    print(f"Results saved to: {path}")

    if run_retrieval:
        run_retrieval_test(comparisons, json_path=path)


# ---------------------------------------------------------------------------
# Mini-Retrieval Test
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set:
    return set(re.findall(r"\b\w{4,}\b", text.lower()))


def _overlap_score(retrieved_text: str, reference: str) -> float:
    ref_words = _word_set(reference)
    if not ref_words:
        return 0.0
    ret_words = _word_set(retrieved_text)
    return round(len(ref_words & ret_words) / len(ref_words), 4)


def run_retrieval_test(comparisons: list, json_path: Path | None = None) -> None:
    """
    Mini-retrieval test: index Level A extracted texts into two temporary in-memory
    Chroma stores (one per method), then run golden set in_scope questions against
    both, comparing hit@3 and word-overlap scores.

    Uses E5 (multilingual-e5-base) embeddings — same model as production.
    No cross-encoder reranking (add-on complexity, not needed for fair comparison).
    No disk I/O: Chroma collections are in-memory only.
    """
    section("MINI-RETRIEVAL TEST")

    # ── lazy imports (heavy; only when --retrieval is passed) ────────────────
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document as LCDocument
    except ImportError as e:
        print(f"  ERROR: missing dependency for retrieval test: {e}")
        print("  Install with: pip install langchain-huggingface langchain-chroma langchain-text-splitters")
        return

    GOLDEN_SET_PATH = PROJECT_ROOT / "evaluation" / "dataset" / "golden_set_it.json"
    if not GOLDEN_SET_PATH.exists():
        print(f"  ERROR: golden set not found at {GOLDEN_SET_PATH}")
        return

    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        gs = json.load(f)
    questions = [
        {"id": q["id"], "question": q["question"], "reference": q["reference"]}
        for q in gs.get("in_scope", [])
        if "reference" in q
    ]
    print(f"  Loaded {len(questions)} in_scope questions from golden set.")

    # ── E5 embedding wrapper (mirrors src/brain.py:47) ───────────────────────
    class _E5Embeddings(HuggingFaceEmbeddings):
        def embed_documents(self, texts):
            return super().embed_documents([f"passage: {t}" for t in texts])
        def embed_query(self, text):
            return super().embed_query(f"query: {text}")

    from config import EMBEDDING_MODEL_NAME, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
    print(f"  Loading embedding model: {EMBEDDING_MODEL_NAME} …")
    emb = _E5Embeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Build document lists from Level A extraction results ─────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    def _build_docs(method: str) -> list:
        docs = []
        for comp in comparisons:
            text = comp.level_a_trafilatura.text if method == "trafilatura" else comp.level_a_crawl4ai.text
            if not text:
                continue
            chunks = splitter.split_text(text)
            for chunk in chunks:
                docs.append(LCDocument(page_content=chunk, metadata={"source": comp.url}))
        return docs

    traf_docs = _build_docs("trafilatura")
    c4ai_docs = _build_docs("crawl4ai")
    print(f"  Trafilatura chunks: {len(traf_docs)} | Crawl4AI chunks: {len(c4ai_docs)}")

    # ── Index into in-memory Chroma stores ───────────────────────────────────
    BATCH = 500
    def _index(docs: list, collection_name: str) -> Chroma:
        store = Chroma(collection_name=collection_name, embedding_function=emb)
        for i in range(0, len(docs), BATCH):
            store.add_documents(docs[i : i + BATCH])
        return store

    print("  Indexing into temporary Chroma stores …")
    traf_store = _index(traf_docs, "tmp_retrieval_traf")
    c4ai_store = _index(c4ai_docs, "tmp_retrieval_c4ai")

    # ── Run queries ──────────────────────────────────────────────────────────
    K = 3
    rows = []
    for q in questions:
        traf_results = traf_store.similarity_search(q["question"], k=K)
        c4ai_results = c4ai_store.similarity_search(q["question"], k=K)

        traf_ctx = " ".join(d.page_content for d in traf_results)
        c4ai_ctx = " ".join(d.page_content for d in c4ai_results)

        traf_overlap = _overlap_score(traf_ctx, q["reference"])
        c4ai_overlap = _overlap_score(c4ai_ctx, q["reference"])

        # hit@K: any retrieved chunk shares ≥5 meaningful words with reference
        ref_words = _word_set(q["reference"])
        traf_hit = any(len(_word_set(d.page_content) & ref_words) >= 5 for d in traf_results)
        c4ai_hit = any(len(_word_set(d.page_content) & ref_words) >= 5 for d in c4ai_results)

        rows.append({
            "id": q["id"],
            "question": q["question"][:60],
            "traf_hit": traf_hit,
            "c4ai_hit": c4ai_hit,
            "traf_overlap": traf_overlap,
            "c4ai_overlap": c4ai_overlap,
        })

    # ── Cleanup stores ───────────────────────────────────────────────────────
    try:
        traf_store.delete_collection()
        c4ai_store.delete_collection()
    except Exception:
        pass

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\n  {'ID':<18} {'Hit@3 Traf':>10} {'Hit@3 C4AI':>10} {'Overlap Traf':>13} {'Overlap C4AI':>13} {'Winner':>8}")
    print("  " + "-" * 78)
    for r in rows:
        winner = "="
        if r["traf_overlap"] > r["c4ai_overlap"] + 0.02:
            winner = "TRAF"
        elif r["c4ai_overlap"] > r["traf_overlap"] + 0.02:
            winner = "C4AI"
        print(
            f"  {r['id']:<18} "
            f"{'YES' if r['traf_hit'] else 'no':>10} "
            f"{'YES' if r['c4ai_hit'] else 'no':>10} "
            f"{r['traf_overlap']:>13.4f} "
            f"{r['c4ai_overlap']:>13.4f} "
            f"{winner:>8}"
        )

    traf_hit_rate = round(sum(r["traf_hit"] for r in rows) / len(rows), 4) if rows else 0.0
    c4ai_hit_rate = round(sum(r["c4ai_hit"] for r in rows) / len(rows), 4) if rows else 0.0
    traf_mean_overlap = round(sum(r["traf_overlap"] for r in rows) / len(rows), 4) if rows else 0.0
    c4ai_mean_overlap = round(sum(r["c4ai_overlap"] for r in rows) / len(rows), 4) if rows else 0.0

    print("  " + "-" * 78)
    print(
        f"  {'AGGREGATE':<18} "
        f"{traf_hit_rate:>10.4f} "
        f"{c4ai_hit_rate:>10.4f} "
        f"{traf_mean_overlap:>13.4f} "
        f"{c4ai_mean_overlap:>13.4f}"
    )

    # ── Append to JSON ───────────────────────────────────────────────────────
    if json_path and json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            data["retrieval_test"] = {
                "n_questions": len(rows),
                "k": K,
                "trafilatura": {"hit_rate": traf_hit_rate, "mean_overlap": traf_mean_overlap},
                "crawl4ai": {"hit_rate": c4ai_hit_rate, "mean_overlap": c4ai_mean_overlap},
                "per_question": rows,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n  Retrieval results appended to: {json_path}")
        except Exception as e:
            print(f"  WARNING: could not update JSON: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Crawl4AI vs trafilatura comparison benchmark")
    parser.add_argument(
        "--retrieval", action="store_true",
        help="Run mini-retrieval test after extraction comparison (requires langchain-huggingface)"
    )
    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)
    asyncio.run(run_comparison(run_retrieval=args.retrieval))


if __name__ == "__main__":
    main()
