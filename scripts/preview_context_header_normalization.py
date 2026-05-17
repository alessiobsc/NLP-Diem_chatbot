"""
Preview contextual-header normalization rules without touching the vector store.

The script reads the JSON produced by scripts/audit_context_headers.py and
simulates two conservative operations:

1. safe formatting that preserves local detail, such as removing square
   brackets without dropping the text inside them;
2. semantic repairs only when the source URL makes the generated header
   semantically wrong.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_IN = PROJECT_ROOT / "evaluation" / "results" / "context_header_audit.json"
DEFAULT_JSON_OUT = PROJECT_ROOT / "evaluation" / "results" / "context_header_normalization_preview.json"
DEFAULT_MD_OUT = PROJECT_ROOT / "evaluation" / "results" / "context_header_normalization_preview.md"

CONTEXT_PREFIX_RE = re.compile(r"^\s*context\s*:\s*", re.IGNORECASE)
BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
SPACES_RE = re.compile(r"\s+")

PLACEHOLDER_PATTERNS = (
    "tipo documento",
    "argomento esplicito",
    "pagina corso di studio",
)

GENERIC_LOCAL_THEMES = {
    "",
    "documento",
    "pagina",
    "pagina istituzionale",
    "pagina corso di studio",
    "profilo docente",
    "docente",
    "docenti",
    "docente/personale",
    "docente/personale - profilo docente",
    "docente - profilo docente",
    "docenti e personale",
    "scheda corso di studio",
    "scheda sua corso di studio",
    "scheda sua",
    "corso di studio",
    "regolamento corso di studio",
    "regolamento",
}

HIGH_CONFIDENCE_URL_PREFIXES = (
    ("__schede-sua", "Scheda SUA corso di studio", "url:schede_sua"),
    ("__regolamenti-cds", "Regolamento corso di studio", "url:regolamenti_cds"),
    ("__piano-studi-cds", "Piano di studi", "url:piano_studi_cds"),
    ("__statistiche-corsi", "Statistiche corso di studio", "url:statistiche_corsi"),
    ("__almalaurea", "Dati AlmaLaurea corso di studio", "url:almalaurea"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview contextual-header normalization rules from an audit JSON."
    )
    parser.add_argument("--audit-in", type=Path, default=DEFAULT_AUDIT_IN)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--max-examples", type=int, default=120)
    return parser.parse_args()


def clean_header_text(header: str, *, remove_placeholders: bool = True) -> str:
    text = CONTEXT_PREFIX_RE.sub("", header or "").strip()

    if "[" in text or "]" in text:
        text = BRACKET_RE.sub(lambda m: m.group(1), text)
        text = text.replace("[", "").replace("]", "")

    if remove_placeholders:
        for placeholder in PLACEHOLDER_PATTERNS:
            text = re.sub(
                rf"(?i)(^|\s*-\s*){re.escape(placeholder)}(\s*-\s*|$)",
                lambda m: " - " if m.group(1).strip("- ") and m.group(2).strip("- ") else "",
                text,
            )

    text = text.replace(" / ", "/")
    text = SPACES_RE.sub(" ", text).strip(" -")
    return text


def normalize_header_format(header: str) -> tuple[str, list[str]]:
    """
    Apply safe formatting only. Do not remove placeholders here, because those
    need source-aware semantic repair.
    """
    original = header or ""
    body = CONTEXT_PREFIX_RE.sub("", original).strip()
    rules: list[str] = []

    if "[" in body or "]" in body:
        body = BRACKET_RE.sub(lambda m: m.group(1), body)
        body = body.replace("[", "").replace("]", "")
        rules.append("format:remove_square_brackets")

    before_spacing = body
    body = body.replace(" / ", "/")
    body = re.sub(r"\s*-\s*", " - ", body)
    body = SPACES_RE.sub(" ", body).strip(" -")
    if body != before_spacing:
        rules.append("format:normalize_spacing")

    before_case = body
    body = normalize_case(body)
    if body:
        body = body[0].upper() + body[1:]
    if body != before_case:
        rules.append("format:normalize_case")

    if has_placeholder(body):
        return original, []

    normalized = f"Context: {body}" if body else original
    if normalized == original:
        return original, []
    return normalized, rules


def normalize_case(text: str) -> str:
    text = SPACES_RE.sub(" ", text or "").strip(" -")
    if not text:
        return ""
    known = {
        "sua": "SUA",
        "diem": "DIEM",
        "cfu": "CFU",
        "phd": "PhD",
        "almalaurea": "AlmaLaurea",
        "erasmus+": "Erasmus+",
        "covid-19": "COVID-19",
        "covid19": "COVID-19",
        "iot": "IoT",
    }
    words = []
    for word in text.split(" "):
        replacement = known.get(word.lower())
        words.append(replacement if replacement else word)
    return " ".join(words)


def split_header_theme(header: str) -> tuple[str, str]:
    if " - " in header:
        left, right = header.split(" - ", 1)
        return left.strip(), right.strip()
    return header.strip(), ""


def has_placeholder(header: str) -> bool:
    lowered = (header or "").lower()
    return any(pattern in lowered for pattern in PLACEHOLDER_PATTERNS)


def contains_docente_profile(header: str) -> bool:
    lowered = clean_header_text(header).lower()
    return (
        "docente/personale" in lowered
        or "profilo docente" in lowered
        or lowered.startswith("docente -")
        or lowered == "docente"
        or "curriculum docente" in lowered
        or "elenco docenti" in lowered
    )


def contains_scheda_insegnamento(header: str) -> bool:
    return "scheda insegnamento" in clean_header_text(header).lower()


def contains_scheda_sua(header: str) -> bool:
    return "scheda sua" in clean_header_text(header).lower()


def compact_theme(theme: str) -> str:
    theme = SPACES_RE.sub(" ", theme or "").strip(" -")
    lowered = theme.lower()
    if lowered in GENERIC_LOCAL_THEMES:
        return ""
    return theme


def infer_topic_from_source(source: str) -> str:
    parsed = urlparse(source or "")
    path = parsed.path.lower()
    query = parse_qs(parsed.query)

    if "avviso" in query or "avvisi" in query:
        return "avviso"
    if path.endswith("/curriculum") or "/curriculum" in path:
        return "curriculum"
    if "/ricerca/pubblicazioni" in path:
        return "pubblicazioni"
    if "/ricerca/progetti" in path:
        return "progetti di ricerca"
    if "/ricerca/laboratori" in path:
        return "laboratori di ricerca"
    if "/ricerca/premi-ricerca" in path:
        return "premi per la ricerca"
    if "/didattica" in path:
        return "didattica"
    if "/international/erasmus" in path:
        return "accordi Erasmus"
    if "/international/" in path:
        return "internazionalizzazione"
    return ""


def high_confidence_prefix(source: str) -> tuple[str, str]:
    lowered = (source or "").lower()
    for marker, prefix, rule in HIGH_CONFIDENCE_URL_PREFIXES:
        if marker in lowered:
            return prefix, rule

    if "dpd-2023-diem" in lowered:
        return "Documento di pianificazione dipartimento", "url:dpd_diem"
    if "/uploads/rescue/499/9017/" in lowered:
        return "Photovoltaics PhD curriculum", "url:photovoltaics_curriculum_pdf"
    return "", ""


def local_theme_for_forced_prefix(prefix: str, cleaned_header: str, source: str) -> str:
    left, right = split_header_theme(cleaned_header)
    lowered = cleaned_header.lower()
    topic = compact_theme(right) or compact_theme(left)

    if prefix == "Scheda SUA corso di studio":
        if "docente" in lowered:
            return "docenti di riferimento"
        if "scheda insegnamento" in lowered:
            return right or "insegnamenti"
        if "mobilità" in lowered or "erasmus" in lowered or "international" in lowered:
            return "mobilità internazionale"
        if "servizi" in lowered or "disabilità" in lowered or "dsa" in lowered:
            return "servizi studenti"
        if "verbale" in lowered or "comitato" in lowered or "parti interessate" in lowered:
            return "consultazioni parti interessate"

    if prefix == "Regolamento corso di studio":
        if "scheda insegnamento" in lowered:
            return f"insegnamento {right}".strip() if right else "insegnamenti"
        if "scheda sua" in lowered:
            return "didattica"

    if prefix == "Dati AlmaLaurea corso di studio":
        if "docent" in lowered:
            return "opinioni sui docenti"
        if "soddisfazione" in lowered:
            return "soddisfazione per il corso di studio"
        if "occup" in lowered:
            return "occupazione laureati"
        if "laureati" in lowered or "statistiche" in lowered:
            return "statistiche laureati"

    if prefix == "Statistiche corso di studio":
        if "iscritt" in lowered:
            return "iscritti"
        if "laureat" in lowered:
            return "laureati"
        if "abbandon" in lowered or "progression" in lowered:
            return "progressioni e abbandoni"

    if prefix == "Photovoltaics PhD curriculum":
        filename = Path(urlparse(source).path).stem.replace("-", " ")
        return filename

    if topic.lower() in {prefix.lower(), "corso di studio"}:
        topic = ""
    return topic


def title_from_filename(source: str) -> str:
    filename = Path(urlparse(source).path).stem.replace("-", " ")
    filename = SPACES_RE.sub(" ", filename).strip()
    return normalize_case(filename)


def schede_sua_topic(cleaned_header: str) -> str:
    lowered = cleaned_header.lower()
    left, right = split_header_theme(cleaned_header)
    topic = compact_theme(right) or compact_theme(left)

    if "rappresentanti studenti" in lowered:
        return "rappresentanti studenti"
    if "docente" in lowered:
        return "docenti di riferimento"
    if "scheda insegnamento" in lowered:
        return compact_theme(right) or "insegnamenti"
    if "mobilità" in lowered or "erasmus" in lowered or "international" in lowered:
        return "mobilità internazionale"
    if "servizi" in lowered or "disabilità" in lowered or "dsa" in lowered:
        return "servizi studenti"
    if "verbale" in lowered or "comitato" in lowered or "parti interessate" in lowered:
        return "consultazioni parti interessate"
    if "parere" in lowered:
        return "parere istituzione corso di studio"
    if "orientamento" in lowered:
        return "orientamento"
    if "didattica" in lowered:
        return "didattica"
    return topic


def regolamento_topic(cleaned_header: str) -> str:
    lowered = cleaned_header.lower()
    left, right = split_header_theme(cleaned_header)
    if "scheda insegnamento" in lowered:
        right = compact_theme(right)
        if right and right.lower() != "scheda insegnamento":
            return f"insegnamento {right}"
        return "insegnamenti"
    if "scheda sua" in lowered:
        right = compact_theme(right)
        left = compact_theme(left)
        if right and "scheda" not in right.lower() and "corso di studio" not in right.lower():
            return right
        if left and "scheda" not in left.lower() and "corso di studio" not in left.lower():
            return left
        return "didattica"
    if "docente" in lowered:
        return "docenti e insegnamenti"
    return compact_theme(right)


def almalaurea_topic(cleaned_header: str) -> str:
    lowered = cleaned_header.lower()
    if "docent" in lowered:
        return "opinioni sui docenti"
    if "soddisfazione" in lowered:
        return "soddisfazione per il corso di studio"
    if "occup" in lowered:
        return "occupazione laureati"
    if "valutazione" in lowered or "opinioni" in lowered:
        return "opinioni laureati"
    if "laureat" in lowered or "statistic" in lowered:
        return "statistiche laureati"
    return ""


def statistiche_topic(cleaned_header: str) -> str:
    lowered = cleaned_header.lower()
    if "iscritt" in lowered or "iscrizion" in lowered:
        return "iscritti"
    if "laureat" in lowered:
        return "laureati"
    if "abbandon" in lowered or "progression" in lowered:
        return "progressioni e abbandoni"
    if "voto" in lowered:
        return "voto medio"
    return ""


def piano_studi_topic(cleaned_header: str) -> str:
    lowered = cleaned_header.lower()
    if "insegn" in lowered or "scheda" in lowered:
        return "insegnamenti"
    if "cfu" in lowered or "credit" in lowered:
        return "crediti formativi"
    return ""


def with_topic(prefix: str, topic: str) -> str:
    topic = normalize_case(compact_theme(topic))
    if topic and topic.lower() != prefix.lower():
        return f"Context: {prefix} - {topic}"
    return f"Context: {prefix}"


def docenti_prefix(source: str) -> tuple[str, str]:
    parsed = urlparse(source or "")
    if parsed.netloc != "docenti.unisa.it":
        return "", ""

    path = parsed.path.lower()
    query = parse_qs(parsed.query)
    if "avviso" in query or "avvisi" in query:
        return "Avviso docente", "url:docenti_avviso"
    if "/curriculum" in path:
        return "Curriculum docente", "url:docenti_curriculum"
    if "/ricerca/pubblicazioni" in path:
        return "Pubblicazioni docente", "url:docenti_pubblicazioni"
    if "/ricerca/progetti" in path:
        return "Progetti di ricerca docente", "url:docenti_progetti"
    if "/ricerca/laboratori" in path:
        return "Laboratori di ricerca docente", "url:docenti_laboratori"
    if "/didattica" in path:
        return "Didattica docente", "url:docenti_didattica"
    if "/international/erasmus" in path:
        return "Accordi Erasmus docente", "url:docenti_erasmus"
    return "", ""


def repair_header_semantics(header: str, source: str) -> dict[str, Any] | None:
    """
    Return a semantic repair only for high-confidence wrong document types.
    Cosmetic-only differences are deliberately ignored.
    """
    cleaned = clean_header_text(header)
    lowered = cleaned.lower()
    source_lower = (source or "").lower()
    placeholder = has_placeholder(header)

    if "__schede-sua" in source_lower:
        if contains_docente_profile(header):
            return {
                "repaired_header": "Context: Scheda SUA corso di studio - docenti di riferimento",
                "rule": "schede_sua:docente_profile",
                "confidence": "high",
                "reason": "URL is __schede-sua but header classifies the chunk as a docente profile.",
            }
        if contains_scheda_insegnamento(header):
            return {
                "repaired_header": with_topic("Scheda SUA corso di studio", schede_sua_topic(cleaned)),
                "rule": "schede_sua:scheda_insegnamento",
                "confidence": "high",
                "reason": "URL is __schede-sua and the local topic is an insegnamento section.",
            }
        if placeholder:
            return {
                "repaired_header": with_topic("Scheda SUA corso di studio", schede_sua_topic(cleaned)),
                "rule": "schede_sua:placeholder",
                "confidence": "high",
                "reason": "URL is __schede-sua and generated header contains a placeholder.",
            }
        if not contains_scheda_sua(header) and any(
            token in lowered
            for token in (
                "mobilità",
                "international",
                "erasmus",
                "servizi",
                "disabilità",
                "dsa",
                "verbale",
                "comitato",
                "parere",
            )
        ):
            return {
                "repaired_header": with_topic("Scheda SUA corso di studio", schede_sua_topic(cleaned)),
                "rule": "schede_sua:missing_document_type",
                "confidence": "medium",
                "reason": "URL is __schede-sua but the header only names a local subsection.",
            }

    if "__regolamenti-cds" in source_lower:
        if contains_scheda_insegnamento(header):
            return {
                "repaired_header": with_topic("Regolamento corso di studio", regolamento_topic(cleaned)),
                "rule": "regolamenti_cds:scheda_insegnamento",
                "confidence": "high",
                "reason": "URL is __regolamenti-cds but header classifies the chunk as a standalone scheda insegnamento.",
            }
        if contains_scheda_sua(header):
            return {
                "repaired_header": with_topic("Regolamento corso di studio", regolamento_topic(cleaned)),
                "rule": "regolamenti_cds:scheda_sua",
                "confidence": "high",
                "reason": "URL is __regolamenti-cds but header classifies the chunk as Scheda SUA.",
            }
        if contains_docente_profile(header):
            return {
                "repaired_header": with_topic("Regolamento corso di studio", regolamento_topic(cleaned)),
                "rule": "regolamenti_cds:docente_profile",
                "confidence": "high",
                "reason": "URL is __regolamenti-cds but header classifies the chunk as a docente profile.",
            }
        if placeholder:
            return {
                "repaired_header": with_topic("Regolamento corso di studio", regolamento_topic(cleaned)),
                "rule": "regolamenti_cds:placeholder",
                "confidence": "high",
                "reason": "URL is __regolamenti-cds and generated header contains a placeholder.",
            }

    if "__almalaurea" in source_lower:
        if contains_docente_profile(header) or contains_scheda_sua(header) or placeholder:
            return {
                "repaired_header": with_topic("Dati AlmaLaurea corso di studio", almalaurea_topic(cleaned)),
                "rule": "almalaurea:wrong_document_type",
                "confidence": "high",
                "reason": "URL is __almalaurea but header uses another document type.",
            }

    if "__statistiche-corsi" in source_lower:
        if "statistic" not in lowered or placeholder or contains_docente_profile(header):
            return {
                "repaired_header": with_topic("Statistiche corso di studio", statistiche_topic(cleaned)),
                "rule": "statistiche_corsi:wrong_document_type",
                "confidence": "high",
                "reason": "URL is __statistiche-corsi but header does not preserve the statistics document type.",
            }

    if "__piano-studi-cds" in source_lower:
        if "piano" not in lowered or placeholder:
            return {
                "repaired_header": with_topic("Piano di studi", piano_studi_topic(cleaned)),
                "rule": "piano_studi_cds:wrong_document_type",
                "confidence": "high",
                "reason": "URL is __piano-studi-cds but header does not preserve the piano di studi document type.",
            }

    if "/uploads/rescue/499/9017/" in source_lower:
        if contains_docente_profile(header) or "phd candidate" in lowered:
            return {
                "repaired_header": with_topic("Photovoltaics PhD curriculum", title_from_filename(source)),
                "rule": "photovoltaics_pdf:docente_profile",
                "confidence": "high",
                "reason": "Photovoltaics curriculum PDF was classified as a docente/candidate profile.",
            }

    return None


def iter_candidates(audit: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    def add(header: str, source: str, count: int, flags: list[str], origin: str, title: str = "") -> None:
        if not header or not source:
            return
        key = (header, source)
        item = candidates.setdefault(
            key,
            {
                "header": header,
                "source": source,
                "title": title,
                "count": 0,
                "flags": sorted(set(flags or [])),
                "origins": set(),
            },
        )
        item["count"] += int(count or 0)
        item["flags"] = sorted(set(item["flags"]) | set(flags or []))
        item["origins"].add(origin)

    for origin in ("needs_manual_review", "top_repeated_headers"):
        for item in audit.get(origin, []):
            header = item.get("header", "")
            flags = item.get("flags", [])
            for source_item in item.get("top_sources", []):
                add(header, source_item.get("source", ""), source_item.get("count", 0), flags, origin)
            for sample in item.get("sample_chunks", []):
                add(header, sample.get("source", ""), 1, flags, f"{origin}:sample", sample.get("title", ""))

    output = []
    for item in candidates.values():
        item["origins"] = sorted(item["origins"])
        output.append(item)
    return sorted(output, key=lambda x: x["count"], reverse=True)


def build_preview(audit: dict[str, Any]) -> dict[str, Any]:
    changes = []
    noops = 0
    for candidate in iter_candidates(audit):
        repair = repair_header_semantics(candidate["header"], candidate["source"])
        if repair and repair["repaired_header"] != candidate["header"]:
            normalized_header = repair["repaired_header"]
            change_type = "semantic_repair"
            rules = [repair["rule"]]
            confidence = repair["confidence"]
            reason = repair["reason"]
        else:
            normalized_header, rules = normalize_header_format(candidate["header"])
            if not rules or normalized_header == candidate["header"]:
                noops += 1
                continue
            change_type = "safe_format"
            confidence = "safe"
            reason = "Formatting-only normalization that preserves the local header detail."

        if normalized_header == candidate["header"]:
            noops += 1
            continue

        changes.append(
            {
                **candidate,
                "normalized_header": normalized_header,
                "change_type": change_type,
                "rules": rules,
                "confidence": confidence,
                "reason": reason,
            }
        )

    type_counts = Counter(item["change_type"] for item in changes)
    rule_counts = Counter(rule for item in changes for rule in item["rules"])
    confidence_counts = Counter(item["confidence"] for item in changes)
    header_counts = Counter(item["normalized_header"] for item in changes)

    return {
        "metadata": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "script": "scripts/preview_context_header_normalization.py",
            "audit_source": str(DEFAULT_AUDIT_IN),
            "mode": "safe_format_plus_semantic_repair",
        },
        "summary": {
            "candidate_pairs": len(changes) + noops,
            "changed_pairs": len(changes),
            "ignored_pairs": noops,
            "estimated_affected_child_chunks": sum(item["count"] for item in changes),
            "change_types": [
                {"change_type": change_type, "count": count}
                for change_type, count in type_counts.most_common()
            ],
            "rules": [{"rule": rule, "count": count} for rule, count in rule_counts.most_common()],
            "confidence": [
                {"confidence": confidence, "count": count}
                for confidence, count in confidence_counts.most_common()
            ],
            "top_normalized_headers": [
                {"header": header, "source_header_pairs": count}
                for header, count in header_counts.most_common(30)
            ],
        },
        "changes": changes,
    }


def one_line(text: str, limit: int = 180) -> str:
    text = SPACES_RE.sub(" ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def write_markdown(report: dict[str, Any], path: Path, max_examples: int) -> None:
    summary = report["summary"]
    lines = [
        "# Context Header Normalization Preview",
        "",
        f"- Generated at: `{report['metadata']['generated_at']}`",
        f"- Candidate header/source pairs: **{summary['candidate_pairs']}**",
        f"- Changed pairs: **{summary['changed_pairs']}**",
        f"- Ignored pairs: **{summary['ignored_pairs']}**",
        f"- Estimated affected child chunks: **{summary['estimated_affected_child_chunks']}**",
        "",
        "## Change Types",
        "",
    ]
    for item in summary["change_types"]:
        lines.append(f"- `{item['change_type']}`: {item['count']}")

    lines.extend([
        "",
        "## Rules Applied",
        "",
    ])
    for item in summary["rules"]:
        lines.append(f"- `{item['rule']}`: {item['count']}")

    lines.extend(["", "## Confidence", ""])
    for item in summary["confidence"]:
        lines.append(f"- `{item['confidence']}`: {item['count']}")

    lines.extend(["", "## Top Repaired Headers", ""])
    for item in summary["top_normalized_headers"][:20]:
        lines.append(f"- {item['source_header_pairs']}: `{item['header']}`")

    lines.extend(["", "## Review Examples", ""])
    for item in report["changes"][:max_examples]:
        lines.extend(
            [
                f"### {item['count']} chunks",
                f"- Source: `{item['source']}`",
                f"- Change type: `{item['change_type']}`",
                f"- Rules: `{', '.join(item['rules'])}`",
                f"- Confidence: `{item['confidence']}`",
                f"- Reason: {item['reason']}",
                f"- Before: `{one_line(item['header'])}`",
                f"- After: `{one_line(item['normalized_header'])}`",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit = json.loads(args.audit_in.read_text(encoding="utf-8"))
    report = build_preview(audit)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(report, args.md_out, args.max_examples)

    summary = report["summary"]
    print(f"Wrote JSON preview: {args.json_out}")
    print(f"Wrote Markdown preview: {args.md_out}")
    print(f"Changed pairs: {summary['changed_pairs']}/{summary['candidate_pairs']}")
    print(f"Estimated affected child chunks: {summary['estimated_affected_child_chunks']}")


if __name__ == "__main__":
    main()
