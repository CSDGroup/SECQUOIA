"""Readers for the TATexp experiment description XML."""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TATParser",
    "extract_channel_comments_from_xml",
    "extract_position_comments_from_xml",
]

_WAVELENGTH_PATH = ".//WavelengthData/WavelengthInformation"
_POSITION_PATH = ".//PositionData/PositionInformation"


def _channel_key(name_attr: str) -> str:
    """Normalise a wavelength ``Name`` attribute to a ``wNN`` key."""
    try:
        return f"w{int(name_attr):02d}"
    except ValueError:
        digits = re.sub(r"\D", "", name_attr) or "0"
        return f"w{digits[:2].zfill(2)}"


def extract_channel_comments_from_xml(xml_path: str | None) -> dict[str, str]:
    """Read per channel comments from a TATexp XML file."""
    mapping: dict[str, str] = {}
    if not xml_path or not os.path.isfile(xml_path):
        return mapping

    try:
        root = ET.parse(xml_path).getroot()
        for wavelength in root.findall(_WAVELENGTH_PATH):
            info = wavelength.find("WLInfo")
            if info is None:
                continue
            name_attr = info.get("Name")
            if not name_attr:
                continue
            mapping[_channel_key(name_attr)] = info.get("Comment", "") or ""
    except (ET.ParseError, OSError):
        pass
    return mapping


def extract_position_comments_from_xml(xml_path: str | None) -> dict[int, str]:
    """Read  position comments from a TATexp XML file."""
    mapping: dict[int, str] = {}
    if not xml_path or not os.path.isfile(xml_path):
        return mapping

    try:
        root = ET.parse(xml_path).getroot()
        for position in root.findall(_POSITION_PATH):
            dimension = position.find("PosInfoDimension")
            if dimension is None:
                continue
            index = dimension.get("index")
            if not index:
                continue
            try:
                position_number = int(index.lstrip("0") or "0")
            except ValueError:
                continue
            mapping[position_number] = dimension.get("comments", "") or ""
    except (ET.ParseError, OSError):
        pass
    return mapping


@dataclass
class TATParser:
    """Parse TATexp.xml and expose searchable metadata.

    Required:
      - MicrometerPerPixel (float)
      - PositionData: index, posX, posY (per index)
      - WavelengthData: channel count and per channel info (name in Comment)
        * width/height must be identical across channels

    Search:
      - query_index(x_um, y_um) -> best-matching position index
        * within any tile’s [x0,x1]×[y0,y1] rect (computed via width/height * µm/px)
        * if multiple tiles contain the point, pick the one with closest centroid.

    """

    # Crucial fields
    um_per_px: float = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)

    # Wavelengths / channels
    channel_count: int = field(init=False)
    wavelengths: list[dict[str, Any]] = field(default_factory=list)

    # Positions
    pos_map: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )  # index -> (posX, posY)
    rects: list[dict[str, Any]] = field(
        default_factory=list
    )  # per index rectangles for search

    # Everything else
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_xml(cls, path: str | bytes | os.PathLike[str]) -> TATParser:
        """Create a parser from a TATexp.xml metadata file."""
        self = cls()
        root = ET.parse(path).getroot()
        if root.tag != "TATSettings":
            raise ValueError("Not a TATexp.xml (root must be <TATSettings>).")

        # 1) Required: MicrometerPerPixel
        um_node = root.find("./MicrometerPerPixel")
        try:
            self.um_per_px = float(um_node.attrib["value"])
        except (AttributeError, KeyError, ValueError, TypeError) as e:
            raise ValueError(
                "MicrometerPerPixel@value must be a valid float."
            ) from e

        # 2) WavelengthData (required width/height consistency + names)
        wl_infos = root.findall(
            ".//WavelengthData/WavelengthInformation/WLInfo"
        )
        if not wl_infos:
            raise ValueError(
                "WavelengthData/WavelengthInformation/WLInfo nodes are required."
            )

        self.wavelengths = []
        widths, heights = set(), set()
        for wl in wl_infos:
            # Channel metadata (name is in "Comment", also keep Name/id)
            entry = dict(wl.attrib)
            entry["ChannelName"] = wl.attrib.get(
                "Comment", ""
            )  # explicit key for convenience
            self.wavelengths.append(entry)

            try:
                w = int(float(wl.attrib["width"]))
                h = int(float(wl.attrib["height"]))
            except (KeyError, ValueError, TypeError) as e:
                raise ValueError(
                    "Each WLInfo must contain numeric 'width' and 'height'."
                ) from e
            widths.add(w)
            heights.add(h)

        if len(widths) != 1 or len(heights) != 1:
            raise ValueError("All channels must share identical width/height.")
        self.width = widths.pop()
        self.height = heights.pop()
        self.channel_count = len(self.wavelengths)

        #  3) Positions (required: index, posX, posY)
        self.pos_map = {}
        for pinf in root.findall(
            ".//PositionData/PositionInformation/PosInfoDimension"
        ):
            idx_s = pinf.attrib.get("index")
            posX_s = pinf.attrib.get("posX")
            posY_s = pinf.attrib.get("posY")
            if idx_s is None or posX_s is None or posY_s is None:
                # skip incomplete entries
                continue
            try:
                idx = int(idx_s)
                posX = float(posX_s)
                posY = float(posY_s)
            except (ValueError, TypeError):
                continue
            self.pos_map[idx] = (posX, posY)

        if not self.pos_map:
            raise ValueError(
                "No valid PositionData entries with index/posX/posY found."
            )

        self._build_rectangles()

        self.extras = _harvest_all_metadata(root)

        return self

    def _build_rectangles(self) -> None:
        """Compute per index coverage rectangles and centroids."""
        dx = self.width * self.um_per_px
        dy = self.height * self.um_per_px
        rects = []
        for idx, (x0, y0) in self.pos_map.items():
            rect = {
                "index": idx,
                "x0": x0,
                "y0": y0,
                "x1": x0 + dx,
                "y1": y0 + dy,
                "cx": x0 + dx / 2.0,
                "cy": y0 + dy / 2.0,
            }
            rects.append(rect)
        self.rects = rects

    def query_index(self, x_um: float, y_um: float) -> int | None:
        """Return the index whose coverage contains (x, y).

        If multiple hit, pick the one with centroid closest to (x, y). If
        none match, return None.
        """
        hits: list[dict[str, Any]] = []
        for r in self.rects:
            if r["x0"] <= x_um <= r["x1"] and r["y0"] <= y_um <= r["y1"]:
                hits.append(r)

        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]["index"]

        # tie-break: closest centroid
        best = min(
            hits, key=lambda r: (r["cx"] - x_um) ** 2 + (r["cy"] - y_um) ** 2
        )
        return best["index"]


def _harvest_all_metadata(root: ET.Element) -> dict[str, Any]:
    """Flatten the XML into a nested dict."""

    def flat(node: ET.Element) -> dict[str, Any]:
        d: dict[str, Any] = {}
        # attributes + text
        if node.attrib:
            for k, v in node.attrib.items():
                d[f"@{k}"] = v
        txt = (node.text or "").strip()
        if txt:
            d["#text"] = txt
        # children
        children = list(node)
        if children:
            bucket: dict[str, list[dict[str, Any]]] = {}
            for ch in children:
                bucket.setdefault(ch.tag, []).append(flat(ch))
            d.update(bucket)
        return d

    return {root.tag: flat(root)}
