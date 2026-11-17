"""Whitelist-based SVG sanitizer for Chubby's drawings."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
MAX_SVG_CHARS = 8000
MAX_ELEMENTS = 200
MAX_DEPTH = 6

ALLOWED_TAGS = {"g", "path", "circle", "rect", "line", "polygon", "ellipse", "polyline"}

_NUM = re.compile(r"^-?\d+(\.\d+)?(e-?\d+)?%?$", re.I)
_NUM_LIST = re.compile(r"^[-0-9.,\s eE+]{1,600}$")
_PATH_D = re.compile(r"^[MmLlHhVvCcSsQqTtAaZz0-9.,\s+\-eE]{1,1500}$")
_TRANSFORM = re.compile(
    r"^(\s*(translate|rotate|scale)\(\s*[-0-9.,\s]{1,40}\)\s*){1,4}$"
)
_COLOR = re.compile(
    r"^(none|#[0-9a-fA-F]{3,8}|[a-zA-Z]{3,20}|"
    r"rgba?\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*(,\s*[0-9.]+\s*)?\))$"
)
_LINECAP = {"butt", "round", "square"}
_LINEJOIN = {"miter", "round", "bevel"}

_NUMERIC_ATTRS = {
    "cx", "cy", "r", "rx", "ry", "x", "y", "x1", "y1", "x2", "y2",
    "width", "height", "stroke-width", "opacity", "fill-opacity", "stroke-opacity",
}
_COLOR_ATTRS = {"fill", "stroke"}


def _valid_attr(name: str, value: str) -> bool:
    if any(bad in value.lower() for bad in ("url(", "javascript", "<", ">", "&", "\\", "@import")):
        return False
    value = value.strip()
    if name in _NUMERIC_ATTRS:
        return bool(_NUM.match(value))
    if name in _COLOR_ATTRS:
        return bool(_COLOR.match(value))
    if name == "d":
        return bool(_PATH_D.match(value))
    if name == "points":
        return bool(_NUM_LIST.match(value))
    if name == "transform":
        return bool(_TRANSFORM.match(value))
    if name == "stroke-linecap":
        return value in _LINECAP
    if name == "stroke-linejoin":
        return value in _LINEJOIN
    return False


_ATTRS = (
    _NUMERIC_ATTRS
    | _COLOR_ATTRS
    | {"d", "points", "transform", "stroke-linecap", "stroke-linejoin"}
)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _render(el: ET.Element, depth: int, counter: list[int]) -> str:
    tag = _local(el.tag)
    if tag not in ALLOWED_TAGS or depth > MAX_DEPTH:
        return ""
    counter[0] += 1
    if counter[0] > MAX_ELEMENTS:
        return ""
    attrs = []
    for key, value in el.attrib.items():
        k = _local(key)
        if k in _ATTRS and "}" not in key and _valid_attr(k, str(value)):
            attrs.append(f'{k}="{str(value).strip()}"')
    attr_txt = (" " + " ".join(attrs)) if attrs else ""
    if tag == "g":
        inner = "".join(_render(c, depth + 1, counter) for c in el)
        return f"<g{attr_txt}>{inner}</g>"
    return f"<{tag}{attr_txt}/>"


def sanitize_svg(svg: str) -> str | None:
    """Return a safe, normalized SVG string, or None if the input is unusable."""
    if not isinstance(svg, str):
        return None
    text = svg.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if not text or len(text) > MAX_SVG_CHARS * 3:
        return None
    # Reject DTDs / entity tricks / processing instructions outright.
    if re.search(r"<!(DOCTYPE|ENTITY)|<\?(?!xml)", text, re.I):
        return None
    text = re.sub(r"^<\?xml[^>]*\?>\s*", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # Strip entity references (keeps parsing safe).
    text = re.sub(r"&(?!#?\w+;)", "", text)
    text = re.sub(r"&#?\w+;", "", text)
    if not re.match(r"^<svg\b", text, re.I):
        return None
    if "xmlns=" not in text.split(">", 1)[0]:
        text = re.sub(r"^<svg\b", f'<svg xmlns="{SVG_NS}"', text, count=1, flags=re.I)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    if _local(root.tag) != "svg":
        return None
    counter = [0]
    body = "".join(_render(c, 1, counter) for c in root)
    if not body:
        return None
    out = (
        f'<svg xmlns="{SVG_NS}" viewBox="0 0 100 100">{body}</svg>'
    )
    if len(out) > MAX_SVG_CHARS:
        return None
    return out
