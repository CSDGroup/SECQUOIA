"""Parsing and normalizing tTt filename tokens."""

from __future__ import annotations

import os
import re
from pathlib import Path

from SECQUOIA.core.ttt_patterns import (
    PatternSet,
    PresetStore,
    compile_patterns,
    load_patterns,
    load_presets,
    preset_sort_key,
    save_presets,
    search_first,
)

_PATTERNS: PatternSet = load_patterns()
_COMPILED, _PATTERN_ERRORS = compile_patterns(_PATTERNS)


def reload_patterns() -> dict[str, str]:
    """Recompile the parsing patterns from disk. Returns any per field errors."""
    global _PATTERNS, _COMPILED, _PATTERN_ERRORS
    _PATTERNS = load_patterns()
    _COMPILED, _PATTERN_ERRORS = compile_patterns(_PATTERNS)
    return _PATTERN_ERRORS


def channel_is_one_indexed() -> bool:
    """Whether a bare channel number (no 'w'/'c' prefix) counts from 1."""
    return _PATTERNS.ch_one_indexed


def list_preset_names() -> list[str]:
    """Every saved pattern name: built-ins first, then custom ones A-Z."""
    return sorted(load_presets().presets, key=preset_sort_key)


def active_preset_name() -> str:
    """The name of the currently active preset."""
    return load_presets().active


def set_active_preset(name: str) -> dict[str, str]:
    """Switch the active preset by name, save it, and reload."""
    store = load_presets()
    if name not in store.presets:
        return {}
    save_presets(PresetStore(active=name, presets=store.presets))
    return reload_patterns()


def normalize_setup(s: str) -> str:
    """Zero-pad a setup number to at least two digits ('7' -> '07')."""
    digits = re.sub(r"\D", "", s or "")
    if not digits:
        return ""
    return f"{int(digits):02d}"


def yymmdd_compact(date_str: str) -> str:
    """Accept 'YYYY-MM-DD', 'YYYYMMDD', 'YY-MM-DD' or 'YYMMDD' and return 'YYMMDD'."""
    s = re.sub(r"[^0-9]", "", (date_str or "").strip())
    if len(s) == 8:
        s = s[2:]
    if not re.fullmatch(r"\d{6}", s):
        return ""
    yy, mm, dd = s[0:2], s[2:4], s[4:6]
    if not (1 <= int(mm) <= 12 and 1 <= int(dd) <= 31):
        return ""
    return f"{yy}{mm}{dd}"


def normalize_initials(s: str) -> str:
    """Strip non-letters, uppercase, and truncate to 4 characters."""
    return re.sub(r"[^A-Za-z]", "", s or "").upper()[:4]


def normalize_pos(pos: str) -> str:
    """Normalize a position token to tTt style like 'p0001', where possible."""
    pos = (pos or "").strip().lower()
    m = re.fullmatch(r"p(\d+)", pos)
    if m:
        return f"p{int(m.group(1)):04d}"
    m = re.fullmatch(r"xy(\d+)", pos)
    if m:
        return f"p{int(m.group(1)):04d}"
    m = re.fullmatch(r"\d+", pos)
    if m:
        return f"p{int(pos):04d}"
    return pos


def _natural_sort_key(label: str) -> list:
    """Split 'A10' into ['a', 10, ''] so it sorts after 'A2', not before."""
    parts = re.split(r"(\d+)", label.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def resolve_positions(raw_labels: list[str]) -> dict[str, str]:
    """Map every distinct raw position token to its final tTt 'p####' suffix."""
    resolved: dict[str, str] = {}
    used_numbers: set[int] = set()

    distinct = list(dict.fromkeys(raw_labels))
    for raw in distinct:
        if not raw:
            resolved[raw] = ""
            continue
        candidate = normalize_pos(raw)
        if re.fullmatch(r"p\d{4}", candidate):
            resolved[raw] = candidate
            used_numbers.add(int(candidate[1:]))

    next_number = 1
    for raw in sorted(
        (r for r in distinct if r not in resolved), key=_natural_sort_key
    ):
        while next_number in used_numbers:
            next_number += 1
        resolved[raw] = f"p{next_number:04d}"
        used_numbers.add(next_number)
        next_number += 1

    return resolved


def normalize_t(t: str) -> str:
    """Normalize time token to 't00001' (5 digits)."""
    t = (t or "").strip().lower()
    m = re.fullmatch(r"t?=?(\d+)", t)
    if not m:
        return t
    return f"t{int(m.group(1)):05d}"


def normalize_z(z: str) -> str:
    """Normalize z token to 'z001' (3 digits)."""
    z = (z or "").strip().lower()
    m = re.fullmatch(r"z?=?(\d+)", z)
    if not m:
        return z
    return f"z{int(m.group(1)):03d}"


def normalize_w(ch: str, one_indexed: bool = False) -> str:
    """Normalize channel token to 'w00' (2 digits)."""
    ch = (ch or "").strip().lower()

    m = re.fullmatch(r"w=?(\d+)", ch)
    if m:
        return f"w{int(m.group(1)):02d}"

    m = re.fullmatch(r"c=?(\d+)", ch)
    if m:
        idx = int(m.group(1)) - 1
        return f"w{idx:02d}" if idx >= 0 else ""

    m = re.fullmatch(r"\d+", ch)
    if m:
        idx = int(ch) - 1 if one_indexed else int(ch)
        return f"w{idx:02d}" if idx >= 0 else ""

    return ch


def yyyymmdd_from_prefix(s: str) -> str:
    """Convert a 6- or 8-digit date prefix to 'YYYY-MM-DD'."""
    if len(s) == 6:
        yy, mm, dd = s[0:2], s[2:4], s[4:6]
        return f"20{yy}-{mm}-{dd}"
    if len(s) == 8:
        yyyy, mm, dd = s[0:4], s[4:6], s[6:8]
        return f"{yyyy}-{mm}-{dd}"
    return ""


def parse_experiment_token_from_filename(filename: str) -> dict:
    """Parse date/initials/setup from a filename such as
    '240323MA35_p0001_t00001_z001_w00.png'."""
    stem = Path(filename).stem
    m = search_first(_COMPILED["exp_token"], stem)
    if not m:
        d = search_first(_COMPILED["date"], stem)
        return {
            "date": yyyymmdd_from_prefix(d.group("date")) if d else "",
            "initials": "",
            "setup": "",
        }
    return {
        "date": yyyymmdd_from_prefix(m.group("date")),
        "initials": m.group("initials").upper(),
        "setup": normalize_setup(m.group("setup")),
    }


def parse_name(filename: str) -> dict:
    """Extract date/initials/pos/t/z/ch tokens from a filename into a dict."""
    name = os.path.splitext(os.path.basename(filename))[0]
    out = {"date": "", "initials": "", "pos": "", "t": "", "z": "", "ch": ""}

    m = search_first(_COMPILED["date"], name)
    if m:
        out["date"] = yyyymmdd_from_prefix(m.group("date"))

    m = search_first(_COMPILED["time"], name)
    if m:
        out["t"] = m.group("t").lower()

    m = search_first(_COMPILED["pos"], name)
    if m:
        out["pos"] = m.group("pos").lower()

    m = search_first(_COMPILED["z"], name)
    if m:
        out["z"] = m.group("z").lower()

    m = search_first(_COMPILED["ch"], name)
    if m:
        out["ch"] = m.group("ch").lower()

    return out
