"""User-editable filename parsing patterns for the tTt data format transformer."""

import dataclasses
import json
import logging
import re
from pathlib import Path

LOG = logging.getLogger(__name__)

DEFAULT_DATE = r"^(?P<date>\d{6,8})[_-]?"
DEFAULT_TIME = r"(?<![A-Za-z])(?P<t>t=?\d{1,5})"
DEFAULT_POS = r"(?<![A-Za-z])(?P<pos>xy\d{1,4}|p\d{1,4})"
DEFAULT_Z = r"(?<![A-Za-z])(?P<z>z=?\d{1,4})"
DEFAULT_CH = r"(?<![A-Za-z])(?P<ch>[cw]=?\d{1,3})"
DEFAULT_EXP_TOKEN = (
    r"^(?P<date>\d{6,8})[_-]?(?P<initials>[A-Za-z]{2,4})(?P<setup>\d{1,3})"
    r"(?![A-Za-z0-9])"
)

REQUIRED_GROUPS = {
    "date": ("date",),
    "time": ("t",),
    "pos": ("pos",),
    "z": ("z",),
    "ch": ("ch",),
    "exp_token": ("date", "initials", "setup"),
}
FIELDS = tuple(REQUIRED_GROUPS)

CONFIG_PATH = Path.home() / ".SECQUOIA" / "ttt_patterns.json"

_LEICA_POS = r"(?P<pos>[^-]+)--"
_ZEISS_OLYMPUS_POS = r"(?<![A-Za-z])S=?(?P<pos>\d{1,4})"


@dataclasses.dataclass(frozen=True)
class PatternSet:

    date: str = DEFAULT_DATE
    time: str = DEFAULT_TIME
    pos: str = DEFAULT_POS
    z: str = DEFAULT_Z
    ch: str = DEFAULT_CH
    exp_token: str = DEFAULT_EXP_TOKEN
    ch_one_indexed: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "PatternSet":
        """Build a PatternSet from a dict, ignoring unknown or wrong-typed keys."""
        if not d:
            return cls()
        field_types = {f.name: f.type for f in dataclasses.fields(cls)}
        return cls(
            **{
                k: v
                for k, v in d.items()
                if k in field_types and isinstance(v, field_types[k])
            }
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


BUILTIN_PRESETS: dict[str, PatternSet] = {
    "Nikon": PatternSet(),
    "Leica": PatternSet(pos=_LEICA_POS),
    "Zeiss/Olympus": PatternSet(pos=_ZEISS_OLYMPUS_POS),
}

FALLBACK_PRESET_NAME = "Nikon"

_BUILTIN_ORDER = {name: i for i, name in enumerate(BUILTIN_PRESETS)}


def is_builtin_preset(name: str) -> bool:
    """Whether ``name`` is one of the permanent, undeletable presets."""
    return name in BUILTIN_PRESETS


def preset_sort_key(name: str) -> tuple:
    """Sort built-in presets first (Nikon, Leica, Zeiss/Olympus), then
    any custom ones alphabetically."""
    if name in _BUILTIN_ORDER:
        return (0, _BUILTIN_ORDER[name])
    return (1, name)


@dataclasses.dataclass(frozen=True)
class PresetStore:
    """Every saved, named pattern set, plus which one is currently active."""

    active: str
    presets: dict[str, PatternSet]

    def active_patterns(self) -> PatternSet:
        return self.presets.get(self.active, PatternSet())


def load_presets(path: Path | str | None = None) -> PresetStore:
    """Load every saved pattern and which one is active."""
    path = Path(path) if path else CONFIG_PATH
    if not path.is_file():
        return PresetStore(
            active=FALLBACK_PRESET_NAME, presets=dict(BUILTIN_PRESETS)
        )

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        LOG.warning(
            "Could not read pattern file %s: %r. Using the defaults.", path, e
        )
        return PresetStore(
            active=FALLBACK_PRESET_NAME, presets=dict(BUILTIN_PRESETS)
        )

    if isinstance(data, dict) and isinstance(data.get("presets"), dict):
        presets = {
            name: PatternSet.from_dict(raw)
            for name, raw in data["presets"].items()
            if isinstance(raw, dict)
        }
        for name, patterns in BUILTIN_PRESETS.items():
            presets.setdefault(name, patterns)
        active = data.get("active")
        if active not in presets:
            active = FALLBACK_PRESET_NAME
        return PresetStore(active=active, presets=presets)

    migrated = PatternSet.from_dict(data)
    return PresetStore(
        active="My patterns",
        presets={"My patterns": migrated, **BUILTIN_PRESETS},
    )


def save_presets(store: PresetStore, path: Path | str | None = None) -> None:
    """Write every saved preset and which one is active, to disk."""
    path = Path(path) if path else CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "active": store.active,
        "presets": {
            name: patterns.to_dict()
            for name, patterns in store.presets.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_patterns(path: Path | str | None = None) -> PatternSet:
    """The currently active pattern set."""
    return load_presets(path).active_patterns()


def compile_field(field: str, raw: str) -> tuple[re.Pattern, str | None]:
    """Compile one field, falling back to its default on any problem."""
    raw = (raw or "").strip("\r\n")
    default_raw = getattr(PatternSet(), field)
    try:
        rx = re.compile(raw, re.IGNORECASE)
    except re.error as e:
        return re.compile(default_raw, re.IGNORECASE), f"invalid regex: {e}"

    missing = [g for g in REQUIRED_GROUPS[field] if g not in rx.groupindex]
    if missing:
        groups = ", ".join(f"(?P<{g}>...)" for g in missing)
        return (
            re.compile(default_raw, re.IGNORECASE),
            f"missing required group(s): {groups}",
        )
    return rx, None


def compile_patterns(
    patterns: PatternSet,
) -> tuple[dict[str, list[re.Pattern]], dict[str, str]]:
    """Compile every field into an ordered list of patterns."""
    defaults = PatternSet()
    compiled: dict[str, list[re.Pattern]] = {}
    errors: dict[str, str] = {}
    for field in FIELDS:
        custom_raw = getattr(patterns, field)
        default_raw = getattr(defaults, field)
        candidates: list[re.Pattern] = []

        if custom_raw.strip() != default_raw.strip():
            rx, error = compile_field(field, custom_raw)
            if error:
                errors[field] = error
            else:
                candidates.append(rx)

        candidates.append(re.compile(default_raw, re.IGNORECASE))
        compiled[field] = candidates
    return compiled, errors


def search_first(patterns: list[re.Pattern], text: str) -> re.Match | None:
    """Search ``text`` with each pattern in order; return the first match."""
    for rx in patterns:
        m = rx.search(text)
        if m:
            return m
    return None


def classify_example(value: str) -> str:
    """Return the regex character class matching ``value``'s shape."""
    if value.isdigit():
        return r"\d"
    if value.isalpha():
        return "[A-Za-z]"
    return "[A-Za-z0-9]"


def build_pattern_from_example(
    group: str, before: str, value: str, after: str
) -> str:
    """Build a search pattern from plain text instead of hand-written regex."""
    before, value, after = (
        before.strip("\r\n"),
        value.strip("\r\n"),
        after.strip("\r\n"),
    )
    width = max(len(value), 1) + 2
    char_class = classify_example(value) if value else r"\d"
    return (
        re.escape(before)
        + f"(?P<{group}>{char_class}{{1,{width}}})"
        + re.escape(after)
    )
