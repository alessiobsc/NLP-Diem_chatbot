"""
Parser comparison: html.parser vs lxml in BeautifulSoup-based link extraction.

Script completamente autocompleto -- non dipende da raw_docs.pkl ne da altri artefatti
di full_comparison o crawler_comparison. Tutte le pagine sono fetched live e messe
in cache locale.

Confronta i tre punti in crawler.py dove BeautifulSoup viene usato:
  A) Sitemap seeds    -- diem.unisa.it e tutti i corsi.unisa.it /?sitemap
  B) Corsi discovery  -- pagine /didattica/offerta-formativa (scoperte via diem sitemap)
  C) Faculty links    -- diem.unisa.it/dipartimento/personale + rubrica pages

Phases:
  --phase fetch    Scarica e mette in cache tutte le pagine processate da BeautifulSoup
  --phase compare  Parsa le pagine cached con entrambi i parser, mostra diff + summary
  --phase all      fetch poi compare

Usage:
  python scripts/parser_comparison.py --phase fetch
  python scripts/parser_comparison.py --phase compare
  python scripts/parser_comparison.py --phase all
"""

import argparse
import hashlib
import json
import re
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
import urllib3
from bs4 import BeautifulSoup
from requests import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.crawler import (
    DIEM_PERSONALE_URL,
    EXCLUDE_DIRS,
    OFFERTA_FORMATIVA_PATH,
    SITEMAP_QUERY,
    build_html_sitemap_url,
    is_pre_2020_url,
)

# -- SSL bypass (mirrors crawler.py) ------------------------------------------
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
_orig_request = Session.request


def _no_ssl_verify(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig_request(self, method, url, **kwargs)


Session.request = _no_ssl_verify

# -- paths ---------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "evaluation" / "parser_comparison"
CACHE_DIR = OUT_DIR / "html_cache"
MANIFEST_PATH = OUT_DIR / "fetch_manifest.json"

PARSERS = ["html.parser", "lxml"]

# Canonical fallback URL per offerta-formativa se non trovata nei seed del sitemap
OFFERTA_FORMATIVA_FALLBACK = "https://www.diem.unisa.it/didattica/offerta-formativa"


# -- cache helpers -------------------------------------------------------------

def _cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    safe = re.sub(r"[^\w.-]", "_", url)[:70]
    return CACHE_DIR / f"{safe}_{h}.html"


def _fetch_and_cache(url: str) -> str | None:
    path = _cache_path(url)
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        resp = requests.get(url, timeout=15, verify=False)
        resp.raise_for_status()
        html = resp.text
        path.write_text(html, encoding="utf-8", errors="replace")
        return html
    except Exception as e:
        print(f"  WARNING fetch failed {url}: {e}")
        return None


# -- extraction helpers (mirror crawler.py, parser-parametric) -----------------

def _extract_sitemap_urls(html: str, sitemap_url: str, base_url: str, parser: str) -> set[str]:
    """Mirror di extract_html_sitemap_urls, parser-parametrico."""
    soup = BeautifulSoup(html, parser)
    base_netloc = urlparse(base_url).netloc
    urls: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:", "tel:")):
            continue
        absolute_url, _ = urldefrag(urljoin(sitemap_url, href))
        parsed = urlparse(absolute_url)
        if parsed.netloc != base_netloc:
            continue
        if SITEMAP_QUERY in absolute_url:
            continue
        if any(exc in absolute_url for exc in EXCLUDE_DIRS):
            continue
        if is_pre_2020_url(absolute_url):
            continue
        urls.add(absolute_url)
    return urls


def _extract_corsi_urls(html: str, parser: str) -> set[str]:
    """Mirror di extract_corsi_urls per singola pagina, parser-parametrico."""
    soup = BeautifulSoup(html, parser)
    corsi: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("http") and "corsi.unisa.it" in href:
            parsed = urlparse(href)
            first_seg = parsed.path.strip("/").split("/")[0]
            if first_seg:
                corsi.add(f"{parsed.scheme}://{parsed.netloc}/{first_seg}")
    return corsi


def _extract_rubrica_urls(html: str, parser: str) -> dict[str, str]:
    """Mirror della parte personale di extract_diem_faculty_urls, parser-parametrico."""
    soup = BeautifulSoup(html, parser)
    result: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("http"):
            continue
        m = re.search(r"matricola=(\d+)", href)
        if m:
            result[m.group(1)] = href
    return result


def _has_docenti_link(html: str, parser: str) -> bool:
    """Mirror della parte rubrica di extract_diem_faculty_urls, parser-parametrico."""
    soup = BeautifulSoup(html, parser)
    return any(
        "docenti.unisa.it" in (a.get("href") or "")
        for a in soup.find_all("a", href=True)
    )


# -- phase fetch ---------------------------------------------------------------

def phase_fetch() -> None:
    _section("PHASE FETCH -- downloading and caching BeautifulSoup-processed pages")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "sitemaps": [],
        "offerta_formativa": [],
        "personale": None,
        "rubrica": [],
    }

    # -- A: DIEM sitemap -------------------------------------------------------
    diem_sm_url = build_html_sitemap_url("https://www.diem.unisa.it/")
    print(f"\n[A] DIEM sitemap: {diem_sm_url}")
    diem_sm_html = _fetch_and_cache(diem_sm_url)
    manifest["sitemaps"].append({
        "url": diem_sm_url,
        "base_url": "https://www.diem.unisa.it/",
        "cache": str(_cache_path(diem_sm_url)),
    })

    # -- B: Discovery offerta-formativa tramite seed del sitemap DIEM ----------
    ofp_urls: list[str] = []
    if diem_sm_html:
        diem_seeds = _extract_sitemap_urls(
            diem_sm_html, diem_sm_url, "https://www.diem.unisa.it/", "html.parser"
        )
        ofp_urls = [
            u for u in diem_seeds
            if OFFERTA_FORMATIVA_PATH in u and "?anno=" not in u
        ]
        print(f"\n[B] Seed del sitemap DIEM: {len(diem_seeds)} totali")
        print(f"  Pagine offerta-formativa trovate nei seed: {len(ofp_urls)}")

    if not ofp_urls:
        print(f"  Nessun seed con {OFFERTA_FORMATIVA_PATH} -- fallback a URL canonica")
        ofp_urls = [OFFERTA_FORMATIVA_FALLBACK]

    all_corsi: set[str] = set()
    for ofp_url in ofp_urls:
        ofp_html = _fetch_and_cache(ofp_url)
        manifest["offerta_formativa"].append({
            "url": ofp_url,
            "cache": str(_cache_path(ofp_url)),
        })
        if ofp_html:
            all_corsi |= _extract_corsi_urls(ofp_html, "html.parser")

    print(f"  Pagine offerta-formativa fetchate: {len(ofp_urls)}")
    print(f"  Course URLs trovati (baseline html.parser): {len(all_corsi)}")

    # -- Sitemaps di ogni corso ------------------------------------------------
    print(f"\n  Fetching sitemaps per {len(all_corsi)} corsi ...")
    for i, corso_url in enumerate(sorted(all_corsi), 1):
        sm_url = build_html_sitemap_url(corso_url)
        _fetch_and_cache(sm_url)
        manifest["sitemaps"].append({
            "url": sm_url,
            "base_url": corso_url,
            "cache": str(_cache_path(sm_url)),
        })
        if i % 10 == 0 or i == len(all_corsi):
            print(f"  [{i}/{len(all_corsi)}] cached")
        time.sleep(0.1)

    # -- C: Personale + rubrica ------------------------------------------------
    print(f"\n[C] DIEM personale: {DIEM_PERSONALE_URL}")
    personale_html = _fetch_and_cache(DIEM_PERSONALE_URL)
    manifest["personale"] = str(_cache_path(DIEM_PERSONALE_URL))

    if personale_html:
        rubrica_map = _extract_rubrica_urls(personale_html, "html.parser")
        print(f"  {len(rubrica_map)} matricole (baseline html.parser)")
        print(f"  Fetching {len(rubrica_map)} rubrica pages ...")
        for i, (mid, rurl) in enumerate(sorted(rubrica_map.items()), 1):
            _fetch_and_cache(rurl)
            manifest["rubrica"].append({
                "matricola": mid,
                "url": rurl,
                "cache": str(_cache_path(rurl)),
            })
            if i % 10 == 0 or i == len(rubrica_map):
                print(f"  [{i}/{len(rubrica_map)}] cached")
            time.sleep(0.3)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Cache:    {CACHE_DIR}")
    print(f"\nRiepilogo fetch:")
    print(f"  Sitemap fetchati:          {len(manifest['sitemaps'])}")
    print(f"  Offerta-formativa fetchate:{len(manifest['offerta_formativa'])}")
    print(f"  Rubrica fetchate:          {len(manifest['rubrica'])}")


# -- phase compare -------------------------------------------------------------

def phase_compare() -> None:
    _section("PHASE COMPARE -- html.parser vs lxml")

    if not MANIFEST_PATH.exists():
        print(f"ERROR: {MANIFEST_PATH} non trovato. Esegui --phase fetch prima.")
        return

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    report: dict = {"timestamp": datetime.now().isoformat(), "sections": {}}

    # -- A: Sitemaps -----------------------------------------------------------
    _section_minor("A. SITEMAP SEEDS (diem.unisa.it + tutti i corsi)")

    sitemap_per_parser: dict[str, set[str]] = {p: set() for p in PARSERS}
    sitemap_rows: list[dict] = []
    missing_cache = 0

    for entry in manifest["sitemaps"]:
        cache_p = Path(entry["cache"])
        if not cache_p.exists():
            missing_cache += 1
            continue
        html = cache_p.read_text(encoding="utf-8", errors="replace")
        sm_url, base_url = entry["url"], entry["base_url"]

        urls_by: dict[str, set[str]] = {}
        for parser in PARSERS:
            try:
                urls_by[parser] = _extract_sitemap_urls(html, sm_url, base_url, parser)
                sitemap_per_parser[parser] |= urls_by[parser]
            except Exception as e:
                urls_by[parser] = set()
                print(f"  ERROR [{parser}] {sm_url}: {e}")

        hp = urls_by["html.parser"]
        lx = urls_by["lxml"]
        sitemap_rows.append({
            "sitemap_url": sm_url,
            "html_parser": len(hp),
            "lxml": len(lx),
            "only_html_parser": sorted(hp - lx),
            "only_lxml": sorted(lx - hp),
        })

    if missing_cache:
        print(f"  WARNING: {missing_cache} pagine non in cache (esegui --phase fetch)")

    print(f"\n  {'Domain/Corso':<52} {'html.parser':>11} {'lxml':>6} {'D':>5}")
    print("  " + "-" * 78)
    pages_with_diff = 0
    for row in sitemap_rows:
        domain = row["sitemap_url"].replace("/?sitemap", "").replace("https://", "")[:52]
        delta = row["lxml"] - row["html_parser"]
        if delta != 0:
            pages_with_diff += 1
        delta_str = f"{delta:+d}" if delta != 0 else "="
        print(f"  {domain:<52} {row['html_parser']:>11} {row['lxml']:>6} {delta_str:>5}")
    print("  " + "-" * 78)

    sm_hp = len(sitemap_per_parser["html.parser"])
    sm_lx = len(sitemap_per_parser["lxml"])
    sm_only_hp = sitemap_per_parser["html.parser"] - sitemap_per_parser["lxml"]
    sm_only_lx = sitemap_per_parser["lxml"] - sitemap_per_parser["html.parser"]
    print(f"  {'TOTALE SEED UNICI':<52} {sm_hp:>11} {sm_lx:>6} {sm_lx - sm_hp:>+5}")
    print(f"\n  Pagine con differenza: {pages_with_diff}/{len(sitemap_rows)}")
    _print_diff_sample("html.parser only", sm_only_hp)
    _print_diff_sample("lxml only", sm_only_lx)

    report["sections"]["sitemaps"] = {
        "html_parser_total": sm_hp,
        "lxml_total": sm_lx,
        "delta": sm_lx - sm_hp,
        "pages_with_diff": pages_with_diff,
        "only_html_parser": sorted(sm_only_hp),
        "only_lxml": sorted(sm_only_lx),
        "per_page": sitemap_rows,
    }

    # -- B: Corsi discovery ----------------------------------------------------
    _section_minor("B. CORSI DISCOVERY (offerta-formativa pages)")

    corsi_per_parser: dict[str, set[str]] = {p: set() for p in PARSERS}
    ofp_entries = manifest.get("offerta_formativa", [])

    if not ofp_entries:
        print("  Nessuna offerta-formativa nel manifest -- sezione B saltata")
    else:
        print(f"  {len(ofp_entries)} pagine offerta-formativa")
        for entry in ofp_entries:
            cache_p = Path(entry["cache"])
            if not cache_p.exists():
                print(f"  MISSING cache: {cache_p.name} -- skipping")
                continue
            html = cache_p.read_text(encoding="utf-8", errors="replace")
            for parser in PARSERS:
                corsi_per_parser[parser] |= _extract_corsi_urls(html, parser)

    co_hp = corsi_per_parser["html.parser"]
    co_lx = corsi_per_parser["lxml"]
    co_only_hp = co_hp - co_lx
    co_only_lx = co_lx - co_hp

    print(f"\n  html.parser: {len(co_hp)} course URL unici")
    print(f"  lxml:        {len(co_lx)} course URL unici")
    print(f"  Delta:       {len(co_lx) - len(co_hp):+d}")
    _print_diff_sample("html.parser only", co_only_hp)
    _print_diff_sample("lxml only", co_only_lx)

    report["sections"]["corsi"] = {
        "html_parser_total": len(co_hp),
        "lxml_total": len(co_lx),
        "delta": len(co_lx) - len(co_hp),
        "only_html_parser": sorted(co_only_hp),
        "only_lxml": sorted(co_only_lx),
    }

    # -- C: Faculty ------------------------------------------------------------
    _section_minor("C. FACULTY LINKS (personale + rubrica pages)")

    matricole_per_parser: dict[str, set[str]] = {p: set() for p in PARSERS}
    personale_cache = manifest.get("personale")
    mp_hp: set[str] = set()
    mp_lx: set[str] = set()

    if personale_cache and Path(personale_cache).exists():
        p_html = Path(personale_cache).read_text(encoding="utf-8", errors="replace")
        for parser in PARSERS:
            rubrica_map = _extract_rubrica_urls(p_html, parser)
            matricole_per_parser[parser] = set(rubrica_map.keys())
        mp_hp = matricole_per_parser["html.parser"]
        mp_lx = matricole_per_parser["lxml"]
        print(f"\n  Personale page -- matricole trovate:")
        print(f"    html.parser: {len(mp_hp)}")
        print(f"    lxml:        {len(mp_lx)}")
        _print_diff_sample("html.parser only", mp_hp - mp_lx)
        _print_diff_sample("lxml only", mp_lx - mp_hp)
    else:
        print("  personale page non in cache")

    validated_per_parser: dict[str, set[str]] = {p: set() for p in PARSERS}
    rubrica_disagreements: list[dict] = []

    for entry in manifest.get("rubrica", []):
        cache_p = Path(entry["cache"])
        if not cache_p.exists():
            continue
        r_html = cache_p.read_text(encoding="utf-8", errors="replace")
        mid = entry["matricola"]
        results = {}
        for parser in PARSERS:
            has = _has_docenti_link(r_html, parser)
            results[parser] = has
            if has:
                validated_per_parser[parser].add(mid)
        if results["html.parser"] != results["lxml"]:
            rubrica_disagreements.append({
                "matricola": mid,
                "url": entry["url"],
                "html_parser": results["html.parser"],
                "lxml": results["lxml"],
            })

    fa_hp = validated_per_parser["html.parser"]
    fa_lx = validated_per_parser["lxml"]
    print(f"\n  Rubrica validation (ha link docenti.unisa.it):")
    print(f"    html.parser: {len(fa_hp)} confermati")
    print(f"    lxml:        {len(fa_lx)} confermati")
    print(f"    Discordanze: {len(rubrica_disagreements)}")
    for d in rubrica_disagreements:
        print(f"      matricola {d['matricola']}: html.parser={d['html_parser']}, lxml={d['lxml']}")

    report["sections"]["faculty"] = {
        "personale": {
            "html_parser": len(mp_hp),
            "lxml": len(mp_lx),
            "only_html_parser": sorted(mp_hp - mp_lx),
            "only_lxml": sorted(mp_lx - mp_hp),
        },
        "rubrica_validated": {
            "html_parser": len(fa_hp),
            "lxml": len(fa_lx),
            "only_html_parser": sorted(fa_hp - fa_lx),
            "only_lxml": sorted(fa_lx - fa_hp),
        },
        "rubrica_disagreements": rubrica_disagreements,
    }

    # -- AGGREGATE -------------------------------------------------------------
    _section("RISULTATO FINALE")

    rows = [
        ("Sitemap seeds",     sm_hp,      sm_lx,      sm_only_hp,    sm_only_lx),
        ("Corsi URLs",        len(co_hp), len(co_lx), co_only_hp,    co_only_lx),
        ("Faculty validated", len(fa_hp), len(fa_lx), fa_hp - fa_lx, fa_lx - fa_hp),
    ]

    print(f"\n  {'Categoria':<24} {'html.parser':>11} {'lxml':>6} {'D':>5} {'solo html.p':>12} {'solo lxml':>10}")
    print("  " + "-" * 74)
    for label, hp_n, lx_n, only_hp_set, only_lx_set in rows:
        delta = lx_n - hp_n
        delta_s = f"{delta:+d}" if delta != 0 else "="
        print(
            f"  {label:<24} {hp_n:>11} {lx_n:>6} {delta_s:>5}"
            f" {len(only_hp_set):>12} {len(only_lx_set):>10}"
        )
    print("  " + "-" * 74)

    overall_diff = sum(abs(lx_n - hp_n) for _, hp_n, lx_n, _, _ in rows)
    if overall_diff == 0:
        print("\n  CONCLUSIONE: nessuna differenza -- html.parser e lxml producono risultati identici.")
    else:
        total_urls = sum(max(hp_n, lx_n) for _, hp_n, lx_n, _, _ in rows)
        print(f"\n  CONCLUSIONE: differenza totale = {overall_diff} URL su {total_urls} unici analizzati.")

    report["aggregate"] = {
        key: {
            "html_parser": hp_n,
            "lxml": lx_n,
            "delta": lx_n - hp_n,
            "only_html_parser": len(only_hp_set),
            "only_lxml": len(only_lx_set),
        }
        for key, (_, hp_n, lx_n, only_hp_set, only_lx_set) in zip(
            ["sitemaps", "corsi", "faculty"], rows
        )
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON salvato: {out_path}")


# -- utility -------------------------------------------------------------------

def _print_diff_sample(label: str, url_set: set | list, n: int = 5) -> None:
    items = sorted(url_set)
    if not items:
        return
    sample = items[:n]
    suffix = f" ... (+{len(items) - n} altri)" if len(items) > n else ""
    print(f"  {label} ({len(items)}): {sample}{suffix}")


def _section(title: str) -> None:
    sep = "=" * 70
    print(f"\n{sep}\n  {title}\n{sep}")


def _section_minor(title: str) -> None:
    print(f"\n{'-' * 60}\n  {title}\n{'-' * 60}")


# -- main ----------------------------------------------------------------------

def main() -> None:
    try:
        import lxml  # noqa: F401
    except ImportError:
        print("ERROR: lxml non installato. Esegui: pip install lxml")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Confronto html.parser vs lxml in BeautifulSoup (crawler UNISA)"
    )
    parser.add_argument(
        "--phase",
        choices=["fetch", "compare", "all"],
        required=True,
        help="fetch: scarica e mette in cache | compare: analizza | all: entrambi",
    )
    args = parser.parse_args()

    if args.phase in ("fetch", "all"):
        phase_fetch()
    if args.phase in ("compare", "all"):
        phase_compare()


if __name__ == "__main__":
    main()
