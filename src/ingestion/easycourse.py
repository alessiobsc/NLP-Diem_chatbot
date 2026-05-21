"""
EasyCourse fetcher for DIEM courses — two data sources:

  Exam calendar (fetch_easycourse_documents):
    combo.php  → catalog with sessions
    test_call.php → all appelli per session
    Output: one LCDocument per (cdl, insegnamento) with ALL 2026 exam slots.

  Lecture schedule (fetch_easycourse_lectures):
    grid_call.php → current week's lectures
    Output: one LCDocument per (cdl, insegnamento) with weekly recurring slots.
    Designed for weekly re-ingestion — each --full run refreshes lecture data.

All endpoints are public (no auth required).
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date

import requests
from langchain_core.documents import Document as LCDocument

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_URL = "https://easycourse.unisa.it/AgendaStudenti"
COMBO_URL = f"{BASE_URL}/combo.php"
EXAM_URL = f"{BASE_URL}/test_call.php"
LECTURE_URL = f"{BASE_URL}/grid_call.php"

REQUEST_TIMEOUT = 20
SLEEP_BETWEEN_CALLS = 0.5

# API uses "06xxx" numeric cdl codes (not "IExxx" display codes).
# IE127→06127, IE128→06128, IE227→06227, IE232→06232, IE233→06233
DIEM_CDL_CODES = {"06127", "06128", "06227", "06232", "06233"}

# Only ingest sessions whose label contains this year string
CURRENT_YEAR_FILTER = "2026"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DIEM-bot/1.0)",
    "Referer": f"{BASE_URL}/index.php",
}


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ExamSlot:
    data: str
    ora_inizio: str
    ora_fine: str
    aula: str
    sede: str
    sessione_label: str
    annullato: bool = False


@dataclass
class ExamEntry:
    cdl_code: str
    corso_nome: str
    insegnamento: str
    crediti: str
    tipo_esame: str
    docente: str
    slots: list[ExamSlot] = field(default_factory=list)


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse(resp: requests.Response):
    """raw_decode stops at first complete JSON value — handles JS var assignments."""
    text = resp.text.strip()
    start = next((i for i, ch in enumerate(text) if ch in "[{"), None)
    if start is None:
        raise ValueError(f"No JSON in response: {text[:200]}")
    obj, _ = json.JSONDecoder().raw_decode(text, start)
    return obj


# ── HTTP ───────────────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(_HEADERS)
    return s


def _fetch_catalog(s: requests.Session) -> list:
    resp = s.get(COMBO_URL, params={"sw": "et_", "page": "corsi"}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return _parse(resp)


def _fetch_exams(s: requests.Session, cdl: str, anno_cdl: str, sessione: str, aaid: str, anno: str) -> dict:
    payload = {"cdl": cdl, "annocdl": anno_cdl, "sessione": sessione, "AAID": aaid, "anno": anno}
    resp = s.post(EXAM_URL, data=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return _parse(resp)


# ── Aggregation ────────────────────────────────────────────────────────────────

def _collect_raw(anno: str = "2025") -> dict[str, ExamEntry]:
    """
    Fetch all exam data and aggregate into {key: ExamEntry} where
    key = f"{cdl_code}|{insegnamento_nome}".
    """
    s = _session()
    entries: dict[str, ExamEntry] = {}
    seen_slots: set[str] = set()

    logger.info("EasyCourse: fetching catalog...")
    try:
        catalog = _fetch_catalog(s)
    except Exception as e:
        logger.error(f"EasyCourse: catalog fetch failed: {e}")
        return {}

    for anno_entry in catalog:
        if str(anno_entry.get("valore", "")) != anno:
            continue

        for corso in anno_entry.get("elenco", []):
            cdl_code = str(corso.get("valore", "")).strip()
            if cdl_code not in DIEM_CDL_CODES:
                continue

            corso_nome = str(corso.get("label", cdl_code))
            logger.info(f"EasyCourse: {cdl_code} — {corso_nome}")

            for anno_corso in corso.get("elenco_anni", []):
                anno_cdl = str(anno_corso.get("valore", "")).strip()
                if anno_cdl.endswith("|-1"):
                    continue  # skip catch-all plan (duplicates)

                for sess in anno_corso.get("elenco_sessioni", []):
                    sess_label = str(sess.get("label", sess.get("nome", "")))
                    if CURRENT_YEAR_FILTER not in sess_label:
                        continue

                    sess_id = str(sess.get("valore", "")).strip()
                    aaid = str(sess.get("AAID", "")).strip()
                    # Use short session name (without date range) for readability
                    sess_short = str(sess.get("nome", sess_label))

                    logger.debug(f"EasyCourse: fetch {cdl_code}/{anno_cdl}/{sess_id}")
                    try:
                        time.sleep(SLEEP_BETWEEN_CALLS)
                        result = _fetch_exams(s, cdl_code, anno_cdl, sess_id, aaid, anno)
                    except Exception as e:
                        logger.warning(f"EasyCourse: fetch failed {cdl_code}/{sess_id}: {e}")
                        continue

                    for ins in result.get("Insegnamenti", []):
                        dati = ins.get("DatiInsegnamento", {})
                        nome = str(dati.get("Nome", "")).strip()
                        if not nome:
                            continue

                        key = f"{cdl_code}|{nome}"
                        if key not in entries:
                            docente = ""
                            # Try to pick docente from first non-empty appello
                            for ap in ins.get("Appelli", []):
                                docente = str(ap.get("docente", ap.get("utilizzatore", ""))).strip()
                                if docente:
                                    break
                            entries[key] = ExamEntry(
                                cdl_code=cdl_code,
                                corso_nome=corso_nome,
                                insegnamento=nome,
                                crediti=str(dati.get("Crediti", "")),
                                tipo_esame=str(dati.get("TipoEsame", "")),
                                docente=docente,
                            )

                        entry = entries[key]

                        for ap in ins.get("Appelli", []):
                            if str(ap.get("Annullato", "0")) == "1":
                                continue

                            data = str(ap.get("Data", "")).strip()
                            ora_i = str(ap.get("OraInizio", "")).strip()
                            ora_f = str(ap.get("OraFine", "")).strip()
                            aula = str(ap.get("Aula", "")).strip()
                            sede = str(ap.get("Sede", "")).strip()

                            slot_key = f"{key}|{data}|{ora_i}"
                            if slot_key in seen_slots:
                                continue
                            seen_slots.add(slot_key)

                            entry.slots.append(ExamSlot(
                                data=data,
                                ora_inizio=ora_i,
                                ora_fine=ora_f,
                                aula=aula,
                                sede=sede,
                                sessione_label=sess_short,
                            ))

    return entries


# ── Formatting ─────────────────────────────────────────────────────────────────

def _entry_to_text(entry: ExamEntry, anno_label: str = "2025/2026") -> str:
    lines = [
        f"Corso: {entry.corso_nome}",
        f"Insegnamento: {entry.insegnamento}",
    ]
    if entry.crediti:
        lines.append(f"CFU: {entry.crediti}")
    if entry.tipo_esame:
        lines.append(f"Tipo esame: {entry.tipo_esame}")
    if entry.docente:
        lines.append(f"Docente: {entry.docente}")
    lines.append(f"Anno accademico: {anno_label}")

    if entry.slots:
        # Sort by date
        sorted_slots = sorted(entry.slots, key=lambda s: s.data)
        lines.append(f"\nAppelli ({len(sorted_slots)}):")
        for slot in sorted_slots:
            ora = f" {slot.ora_inizio}–{slot.ora_fine}" if slot.ora_inizio else ""
            location = slot.aula
            if slot.sede and slot.sede.lower() not in slot.aula.lower():
                location = f"{slot.aula} ({slot.sede})"
            lines.append(f"  - {slot.data}{ora} | {location} | {slot.sessione_label}")
    else:
        lines.append("\nAppelli: nessun appello disponibile per le sessioni 2026")

    return "\n".join(lines)


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_easycourse_documents(anno: str = "2025") -> list[LCDocument]:
    """
    Fetch 2026 exam calendar for all DIEM courses.

    Returns one LCDocument per (cdl, insegnamento) with all exam slots listed,
    so a single retrieval gives the full exam schedule for that course.

    Args:
        anno: Academic year key for POST params ("2025" = 2025/2026).
    """
    anno_label = f"{anno}/{int(anno)+1}"
    entries = _collect_raw(anno)

    docs: list[LCDocument] = []
    for entry in entries.values():
        if not entry.slots:
            continue  # skip courses with no scheduled exams

        text = _entry_to_text(entry, anno_label)
        source_url = f"{BASE_URL}/index.php?view=easytest&cdl={entry.cdl_code}&anno={anno}"
        docs.append(LCDocument(
            page_content=text,
            metadata={
                "source": source_url,
                "context_header": f"calendario esami {entry.insegnamento} - {entry.corso_nome}",
                "title": f"Appelli {entry.insegnamento} ({entry.corso_nome}) {anno_label}",
                "cdl": entry.cdl_code,
                "anno": anno,
            },
        ))

    logger.info(f"EasyCourse: {len(docs)} exam documents produced ({len(entries)} insegnamenti)")
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# LECTURE SCHEDULE  (current week → recurring weekly timetable)
# ══════════════════════════════════════════════════════════════════════════════

# Numeric giorno (1=Mon…7=Sun) → Italian day name
_GIORNO_IT = {
    "1": "Lunedì", "2": "Martedì", "3": "Mercoledì",
    "4": "Giovedì", "5": "Venerdì", "6": "Sabato", "7": "Domenica",
}


def _fetch_lecture_catalog(s: requests.Session, anno: str) -> list:
    """
    Fetch lecture course catalog using the ec_ combo endpoint (separate from
    exam catalog et_). Returns elenco_corsi list with elenco_anni per corso.
    """
    s.get(COMBO_URL, params={"sw": "ec_", "aa": "1"}, timeout=REQUEST_TIMEOUT)
    resp = s.get(COMBO_URL, params={"sw": "ec_", "aa": anno, "page": "corsi"}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return _parse(resp)


def _fetch_week_lectures(
    s: requests.Session,
    cdl_code: str,
    anno2_list: list[str],
    anno: str,
    ref_date: str,
) -> list[dict]:
    """
    POST grid_call.php with ALL anno2 values for one CDL in a single request
    (mirrors the browser form behaviour — one call per CDL, not per year group).
    Returns raw cella list.
    """
    # requests accepts a list of (key, value) tuples for repeated params
    payload: list[tuple[str, str]] = [
        ("view", "easycourse"),
        ("form-type", "corso"),
        ("include", "corso"),
        ("anno", anno),
        ("corso", cdl_code),
        ("visualizzazione_orario", "cal"),
        ("date", ref_date),
        ("all_events", "0"),
        ("week_grid_type", "-1"),
        ("col_cells", "0"),
        ("empty_box", "0"),
        ("only_grid", "0"),
        ("highlighted_date", "0"),
        ("faculty_group", "0"),
    ]
    for a2 in anno2_list:
        payload.append(("anno2[]", a2))

    try:
        resp = s.post(LECTURE_URL, data=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("celle", [])
    except Exception as e:
        logger.warning(f"EasyCourse lectures: grid_call failed {cdl_code}: {e}")
        return []


def fetch_easycourse_lectures(anno: str = "2025") -> list[LCDocument]:
    """
    Fetch the current week's lecture schedule for all DIEM courses.

    Uses the ec_ lecture catalog (distinct from the et_ exam catalog) to
    obtain correct anno2 values, then fires ONE grid_call.php POST per CDL
    with all year-groups bundled — exactly as the browser form does.

    Cells expose top-level fields (nome_insegnamento, giorno, ora_inizio,
    ora_fine, aula, docente) so no HTML parsing is needed.

    Groups slots by (cdl, insegnamento) and produces one LCDocument per
    insegnamento. Designed for weekly re-ingestion.

    Args:
        anno: Academic year key ("2025" = 2025/2026).
    """
    anno_label = f"{anno}/{int(anno)+1}"
    ref_date = date.today().strftime("%d-%m-%Y")

    s = _session()
    s.headers["Referer"] = f"{BASE_URL}/index.php?view=easycourse&_lang=it&include=corso"

    logger.info(f"EasyCourse lectures: fetching current week ({ref_date}) for {anno_label}...")

    try:
        catalog = _fetch_lecture_catalog(s, anno)
    except Exception as e:
        logger.error(f"EasyCourse lectures: catalog fetch failed: {e}")
        return []

    # key → {insegnamento, corso_nome, cdl_code, docente, slots: [(giorno_num, ora_i, ora_f, aula)]}
    entries: dict[str, dict] = {}
    seen_slots: set[str] = set()

    for corso in catalog:
        cdl_code = str(corso.get("valore", "")).strip()
        if cdl_code not in DIEM_CDL_CODES:
            continue

        corso_nome = str(corso.get("label", cdl_code))
        anno2_list = [
            str(a.get("valore", "")).strip()
            for a in corso.get("elenco_anni", [])
            if str(a.get("valore", "")).strip()
        ]
        if not anno2_list:
            continue

        logger.info(f"EasyCourse lectures: {cdl_code} — {corso_nome} ({len(anno2_list)} year groups)")
        time.sleep(SLEEP_BETWEEN_CALLS)
        celle = _fetch_week_lectures(s, cdl_code, anno2_list, anno, ref_date)
        logger.debug(f"EasyCourse lectures: {cdl_code} -> {len(celle)} celle")

        for cella in celle:
            if str(cella.get("Annullato", "0")) == "1":
                continue

            nome = str(cella.get("nome_insegnamento", "")).strip()
            giorno_num = str(cella.get("giorno", "")).strip()
            ora_i = str(cella.get("ora_inizio", "")).strip()
            ora_f = str(cella.get("ora_fine", "")).strip()
            aula = str(cella.get("aula", "")).strip()
            docente = str(cella.get("docente", "")).strip()

            if not nome or not giorno_num or not ora_i:
                continue

            giorno_label = _GIORNO_IT.get(giorno_num, giorno_num)
            key = f"{cdl_code}|{nome}"

            if key not in entries:
                entries[key] = {
                    "cdl_code": cdl_code,
                    "corso_nome": corso_nome,
                    "insegnamento": nome,
                    "docente": docente,
                    "slots": [],
                }
            elif docente and not entries[key]["docente"]:
                entries[key]["docente"] = docente

            slot_key = f"{key}|{giorno_num}|{ora_i}"
            if slot_key not in seen_slots:
                seen_slots.add(slot_key)
                entries[key]["slots"].append((giorno_num, giorno_label, ora_i, ora_f, aula))

    docs: list[LCDocument] = []
    for entry in entries.values():
        if not entry["slots"]:
            continue

        sorted_slots = sorted(entry["slots"], key=lambda s: (int(s[0]), s[2]))

        lines = [
            f"Corso: {entry['corso_nome']}",
            f"Insegnamento: {entry['insegnamento']}",
        ]
        if entry["docente"]:
            lines.append(f"Docente: {entry['docente']}")
        lines.append(f"Anno accademico: {anno_label}")
        lines.append(f"Settimana di riferimento: {ref_date}")
        lines.append(f"\nOrario settimanale ({len(sorted_slots)} slot):")
        for _gnum, giorno_label, ora_i, ora_f, aula in sorted_slots:
            ora = f"{ora_i}–{ora_f}" if ora_f else ora_i
            lines.append(f"  - {giorno_label} {ora}{f' | {aula}' if aula else ''}")

        text = "\n".join(lines)
        source_url = f"{BASE_URL}/index.php?view=easycourse&corso={entry['cdl_code']}&aa={anno}"
        docs.append(LCDocument(
            page_content=text,
            metadata={
                "source": source_url,
                "context_header": f"orario lezioni {entry['insegnamento']} - {entry['corso_nome']}",
                "title": f"Orario {entry['insegnamento']} ({entry['corso_nome']}) {anno_label}",
                "cdl": entry["cdl_code"],
                "anno": anno,
                "ref_date": ref_date,
            },
        ))

    logger.info(f"EasyCourse lectures: {len(docs)} lecture documents produced")
    return docs
