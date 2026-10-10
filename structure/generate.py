#!/usr/bin/env python3
"""Generate figures of the local AI server setup of ../docker-compose.yml.

- agx-ai-structure-specs.svg: with the hostnames and the hardware specifications of the computers
- agx-ai-structure.svg: without the hostnames and the hardware specifications, except for the GPUs

The logos in ./logos are embedded as data URIs, so that each figure is a single self-contained file
that works also in an <img> tag (e.g. in reveal.js). The figures are also rendered to PNG
with headless Chrome or Chromium. Run with: python3 generate.py
"""

import base64
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import pathlib
import re
import shutil
import struct
import subprocess
from xml.sax.saxutils import escape
import zlib

DIR = pathlib.Path(__file__).resolve().parent
LOGOS = DIR / "logos"
OUT = DIR / "agx-ai-structure.svg"
OUT_SPECS = DIR / "agx-ai-structure-specs.svg"
PNG_SCALE = 2  # Device scale factor of the PNG
# Headless Chrome may reserve part of the window height for the browser UI (87 px in Chrome 130),
# so the PNG is rendered with a taller window and then cropped to the height of the figure.
PNG_EXTRA_HEIGHT = 200

WIDTH = 1600
HEIGHT = 900
FONT = "'Source Sans Pro', 'Source Sans 3', Helvetica, Arial, sans-serif"

# Colors
INK = "#1f2933"
MUTED = "#52606d"
LINE = "#52606d"
STACK_FILL = "#eef4fb"
STACK_STROKE = "#7b9cc4"
CLOUD_FILL = "#fdf3ea"
CLOUD_STROKE = "#e0a36f"
HW_FILL = "#f1f7ee"
HW_STROKE = "#8fb67f"
LLAMA = "#ff8236"
MODEL = "#b4531c"


def _svg_data(path: pathlib.Path) -> tuple[bytes, float]:
    text = path.read_text(encoding="utf-8")
    root_match = re.search(r"<svg\b[^>]*>", text, re.S)
    if root_match is None:
        raise ValueError(f"No <svg> element in {path}")
    root = root_match.group(0)
    view_box = re.search(r'viewBox="([^"]+)"', root)
    if view_box:
        _, _, w, h = (float(v) for v in view_box.group(1).replace(",", " ").split())
    else:
        # Without a viewBox, an SVG image is not scaled to the size of the <image> element.
        width = re.search(r'\swidth="([\d.]+)', root)
        height = re.search(r'\sheight="([\d.]+)', root)
        if width is None or height is None:
            raise ValueError(f"The SVG {path} has neither a viewBox nor a width and height")
        w = float(width.group(1))
        h = float(height.group(1))
        text = text.replace(root, root[:-1] + f' viewBox="0 0 {w:g} {h:g}">', 1)
    return text.encode("utf-8"), w / h


def _png_data(path: pathlib.Path) -> tuple[bytes, float]:
    data = path.read_bytes()
    w, h = struct.unpack(">II", data[16:24])
    return data, w / h


def load_logo(name: str) -> tuple[str, float]:
    """Return the data URI and the aspect ratio (width / height) of a logo."""
    path = LOGOS / name
    if path.suffix == ".svg":
        data, aspect = _svg_data(path)
        mime = "image/svg+xml"
    else:
        data, aspect = _png_data(path)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}", aspect


@dataclass
class Machine:
    """A device in the figure with its specifications, models and runtimes."""

    kind: str  # laptop, desktop or server
    cx: float  # Center x coordinate
    name: tuple[str, bool]  # (title, public), the hostname is shown only in the specs version
    lines: list[tuple[str, bool]]  # (text, public) of the hardware and software, see spec() and pub()
    models: list[tuple[str, str, str]]  # (model, generation t/s, prompt processing t/s), the speeds may be empty
    screen: list[str]  # Logos of the operating system and the virtualization, shown on the screen of the device
    runtimes: list[tuple[str, str, str]]  # (logo, title, detail) of what runs on the machine

    def public(self) -> Machine:
        """Return the machine without the hostname and the hardware specifications, except for the GPUs."""
        return Machine(
            self.kind, self.cx, self.name if self.name[1] else ("", True), [line for line in self.lines if line[1]],
            self.models, self.screen, self.runtimes,
        )

    def text_lines(self) -> int:
        """Return the number of text lines below the device, excluding the runtime tags."""
        return (1 if self.name[0] else 0) + len(self.lines) + len(self.models)


def spec(item: str) -> tuple[str, bool]:
    """A title or line shown only in the specs version, e.g. a hostname, the CPU or the RAM."""
    return item, False


def pub(item: str) -> tuple[str, bool]:
    """A title or line shown in both versions, e.g. a laptop model or a GPU."""
    return item, True


class Figure:
    """An SVG figure that is built from drawing primitives and collects the used logos."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.logos: dict[str, tuple[str, float]] = {}

    def add(self, s: str) -> None:
        """Append an SVG fragment to the figure."""
        self.parts.append(s)

    # Primitives

    def _logo_id(self, name: str) -> str:
        """Return the <symbol> id of a logo file."""
        return "logo-" + name.rsplit(".", 1)[0]

    def logo_width(self, name: str, h: float) -> float:
        """Return the natural width of a logo at the height h, loading the logo if needed."""
        if name not in self.logos:
            self.logos[name] = load_logo(name)
        return h * self.logos[name][1]

    def logo(self, name: str, x: float, y: float, h: float, w: float | None = None) -> float:
        """Draw a logo centered in the box (x, y, w, h) and return w.

        By default, w is the natural width at the height h.

        Each logo is embedded only once as a <symbol>, which is then referenced with <use>.
        """
        if w is None:
            w = self.logo_width(name, h)
        else:
            self.logo_width(name, h)
        self.add(f'<use href="#{self._logo_id(name)}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}"/>')
        return w

    def symbols(self) -> str:
        """Return the <symbol> definitions of all used logos."""
        out = []
        for name, (uri, aspect) in self.logos.items():
            w = 100 * aspect
            out.append(
                f'<symbol id="{self._logo_id(name)}" viewBox="0 0 {w:g} 100">'
                f'<image width="{w:g}" height="100" href="{uri}"/></symbol>'
            )
        return "\n".join(out)

    def text(
        self, x: float, y: float, s: str, size: float = 14, weight: str = "normal", color: str = INK,
        anchor: str = "start", style: str = "",
    ) -> None:
        """Draw a text, optionally with a font style."""
        self.add(
            f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}"{f" font-style={chr(34)}{style}{chr(34)}" if style else ""}>{escape(s)}</text>'
        )

    def rect(
        self, x: float, y: float, w: float, h: float, fill: str = "#ffffff", stroke: str = LINE, rx: float = 10,
        width: float = 1.5, dash: str | None = None, extra: str = "",
    ) -> None:
        """Draw a rectangle."""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash_attr}{extra}/>'
        )

    def arrow(self, points: Iterable[tuple[float, float]], dashed: bool = False) -> None:
        """Draw a polyline with an arrowhead at the end."""
        d = "M" + " L".join(f"{x:g},{y:g}" for x, y in points)
        dash_attr = ' stroke-dasharray="7 5"' if dashed else ""
        self.add(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="2"{dash_attr} marker-end="url(#arrow)"/>')

    # Components

    def card(
        self, x: float, y: float, w: float, h: float, logo: str | Callable[[float, float, float], None] | None,
        title: str, subtitle: str | None = None, logo_size: float = 44, fill: str = "#ffffff",
        stroke: str = STACK_STROKE,
    ) -> None:
        """Draw a box with a logo, a title and an optional subtitle."""
        self.rect(x, y, w, h, fill=fill, stroke=stroke)
        pad = (h - logo_size) / 2
        if callable(logo):
            logo(x + 12, y + pad, logo_size)
        elif logo:
            self.logo(logo, x + 12, y + pad, logo_size, logo_size)
        tx = x + 12 + logo_size + 12 if logo else x + 14
        if subtitle:
            self.text(tx, y + h / 2 - 3, title, size=18, weight="bold")
            self.text(tx, y + h / 2 + 16, subtitle, size=13, color=MUTED)
        else:
            self.text(tx, y + h / 2 + 6, title, size=18, weight="bold")

    def terminal_icon(self, x: float, y: float, s: float) -> None:
        """Open Terminal has no logo of its own, so draw a generic terminal icon."""
        self.rect(x, y + s * 0.1, s, s * 0.8, fill="#1f2933", stroke="#1f2933", rx=6)
        self.add(
            f'<path d="M{x + s * 0.18:g},{y + s * 0.32:g} L{x + s * 0.38:g},{y + s * 0.5:g} '
            f'L{x + s * 0.18:g},{y + s * 0.68:g}" fill="none" stroke="#ffffff" stroke-width="3.5" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        self.add(
            f'<path d="M{x + s * 0.45:g},{y + s * 0.7:g} L{x + s * 0.78:g},{y + s * 0.7:g}" '
            f'stroke="#ffffff" stroke-width="3.5" stroke-linecap="round"/>'
        )

    def key_icon(self, x: float, y: float, s: float) -> None:
        """OpenSSH has no official logo, so draw a key."""
        c = "#1f2933"
        r = s * 0.2
        cx, cy = x + s * 0.3, y + s * 0.5
        self.add(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="none" stroke="{c}" stroke-width="4"/>')
        self.add(
            f'<path d="M{cx + r:g},{cy:g} L{x + s * 0.92:g},{cy:g} '
            f'M{x + s * 0.8:g},{cy:g} L{x + s * 0.8:g},{cy + s * 0.17:g} '
            f'M{x + s * 0.68:g},{cy:g} L{x + s * 0.68:g},{cy + s * 0.13:g}" fill="none" stroke="{c}" '
            f'stroke-width="4" stroke-linecap="round"/>'
        )

    def person_icon(self, x: float, y: float, s: float, color: str = MUTED) -> None:
        """Draw a generic person icon."""
        self.add(f'<circle cx="{x + s / 2:g}" cy="{y + s * 0.28:g}" r="{s * 0.2:g}" fill="{color}"/>')
        self.add(
            f'<path d="M{x + s * 0.1:g},{y + s * 0.95:g} C{x + s * 0.1:g},{y + s * 0.55:g} {x + s * 0.9:g},'
            f'{y + s * 0.55:g} {x + s * 0.9:g},{y + s * 0.95:g} Z" fill="{color}"/>'
        )

    # Devices. (cx, top) is the top center of the drawing, which is 150 px wide and 110 px tall.
    # The screen shows the logos of the operating system and the virtualization.

    def screen_logos(self, cx: float, cy: float, logos: list[str], size: float, gap: float = 8) -> None:
        """Draw the logos centered at (cx, cy), stacked vertically."""
        y = cy - (size * len(logos) + gap * (len(logos) - 1)) / 2
        for logo in logos:
            self.logo(logo, cx - size / 2, y, size, size)
            y += size + gap

    def laptop(self, cx: float, top: float, screen: list[str]) -> None:
        """Draw a laptop showing the screen logos."""
        self.rect(cx - 68, top + 6, 136, 88, fill="#3e4c59", stroke="#1f2933", rx=7, width=2)
        self.rect(cx - 60, top + 13, 120, 74, fill="#dbe7f3", stroke="none", rx=2, width=0)
        self.add(
            f'<path d="M{cx - 82:g},{top + 97:g} L{cx + 82:g},{top + 97:g} L{cx + 76:g},{top + 108:g} '
            f'L{cx - 76:g},{top + 108:g} Z" fill="#9aa5b1" stroke="#1f2933" stroke-width="2" stroke-linejoin="round"/>'
        )
        self.rect(cx - 16, top + 97, 32, 4, fill="#7b8794", stroke="none", rx=2, width=0)
        self.screen_logos(cx, top + 50, screen, 56)

    def desktop(self, cx: float, top: float, screen: list[str]) -> None:
        """Draw a desktop computer showing the screen logos."""
        # Monitor
        mx = cx - 40
        self.rect(mx - 72, top, 144, 94, fill="#3e4c59", stroke="#1f2933", rx=7, width=2)
        self.rect(mx - 64, top + 8, 128, 76, fill="#dbe7f3", stroke="none", rx=2, width=0)
        self.add(
            f'<path d="M{mx - 10:g},{top + 94:g} L{mx + 10:g},{top + 94:g} L{mx + 14:g},{top + 108:g} '
            f'L{mx - 14:g},{top + 108:g} Z" fill="#9aa5b1" stroke="#1f2933" stroke-width="2" stroke-linejoin="round"/>'
        )
        self.rect(mx - 36, top + 106, 72, 5, fill="#9aa5b1", stroke="#1f2933", rx=2, width=2)
        self.screen_logos(mx, top + 46, screen, 58)
        # Tower
        tx = cx + 50
        self.rect(tx, top + 6, 50, 105, fill="#3e4c59", stroke="#1f2933", rx=5, width=2)
        for i in range(3):
            self.rect(tx + 8, top + 16 + i * 10, 34, 5, fill="#7b8794", stroke="none", rx=1.5, width=0)
        self.add(f'<circle cx="{tx + 25:g}" cy="{top + 90:g}" r="5" fill="none" stroke="#9aa5b1" stroke-width="2"/>')

    def server(self, cx: float, top: float, screen: list[str]) -> None:
        """Draw a server rack showing the screen logos."""
        # A rack with three server units and a front display.
        self.rect(cx - 80, top, 160, 111, fill="#1f2933", stroke="#1f2933", rx=5, width=2)
        for y in (top + 6, top + 40, top + 74):
            self.rect(cx - 74, y, 148, 31, fill="#3e4c59", stroke="#52606d", rx=3, width=1)
            for j in range(5):
                self.rect(cx - 66 + j * 9, y + 7, 4, 17, fill="#616e7c", stroke="none", rx=1, width=0)
            self.add(f'<circle cx="{cx + 62:g}" cy="{y + 10:g}" r="3.2" fill="#3ebd93"/>')
            self.add(f'<circle cx="{cx + 62:g}" cy="{y + 21:g}" r="3.2" fill="{LLAMA}"/>')
        # Front display
        self.rect(cx - 12, top + 2, 58, 107, fill="#dbe7f3", stroke="#9aa5b1", rx=4, width=1.5)
        self.screen_logos(cx + 17, top + 55.5, screen, 46 if len(screen) == 1 else 40)

    def runtime_tag(self, cx: float, y: float, logo: str, title: str, detail: str) -> None:
        """A pill that shows what runs on a device, e.g. llama.cpp with CUDA.

        A wordmark logo replaces the title. The text width is estimated for Arial-like fonts,
        as an SVG in an <img> tag can use only the fonts installed on the system.
        """
        wordmark = self.logo_width(logo, 1) > 1.5
        logo_w = self.logo_width(logo, 18) if wordmark else 22
        text_w = 0.55 * 14 * (len(detail) + (0 if wordmark else len(title) + 1))
        w = 6 + logo_w + 8 + text_w + 14
        x = cx - w / 2
        stroke = LLAMA if logo == "llama-cpp.svg" else "#6b4fbb"
        self.rect(x, y, w, 30, fill="#ffffff", stroke=stroke, rx=15, width=1.5)
        if wordmark:
            self.logo(logo, x + 10, y + 6, 18, logo_w)
            self.text(x + 10 + logo_w + 8, y + 20, detail, size=14, color=MUTED)
        else:
            self.logo(logo, x + 6, y + 4, 22, 22)
            self.add(
                f'<text x="{x + 34:g}" y="{y + 20:g}" font-size="14" fill="{INK}">'
                f'<tspan font-weight="bold">{escape(title)}</tspan> '
                f'<tspan fill="{MUTED}">{escape(detail)}</tspan></text>'
            )

    def machine(self, m: Machine, top: float, text_lines: int) -> None:
        """Draw a machine.

        The runtime tags are aligned between the machines,
        so the text block has room for text_lines lines below the device.
        """
        getattr(self, m.kind)(m.cx, top, m.screen)
        y = top + 116
        if m.name[0]:
            y += 18
            self.text(m.cx, y, m.name[0], size=18, weight="bold", anchor="middle")
        for line, _ in m.lines:
            y += 18
            self.text(m.cx, y, line, size=14, color=MUTED, anchor="middle")
        for model, *speeds in m.models:
            y += 18
            speed_text = " | ".join(speed for speed in speeds if speed)
            speed_span = f'<tspan font-weight="normal"> ({escape(speed_text)})</tspan>' if speed_text else ""
            self.add(
                f'<text x="{m.cx:g}" y="{y:g}" font-size="14" font-weight="bold" fill="{MODEL}" '
                f'text-anchor="middle">{escape(model)}{speed_span}</text>'
            )
        y = top + 116 + 18 * text_lines + 20
        for logo, title, detail in m.runtimes:
            self.runtime_tag(m.cx, y, logo, title, detail)
            y += 35

    @staticmethod
    def machine_height(text_lines: int, runtimes: int) -> float:
        """Return the height of a machine drawn by machine().

        The height is measured from the top of the device to the bottom of the last runtime tag.
        """
        return 116 + 18 * text_lines + 20 + 35 * runtimes - 5

    def render(self) -> str:
        """Return the complete SVG document."""
        head = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" \
width="{WIDTH}" height="{HEIGHT}" font-family="{FONT}">
<title>Structure of the agx-ai local AI server</title>
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 Z" fill="{LINE}"/>
</marker>
{self.symbols()}
</defs>
<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>
"""
        return head + "\n".join(self.parts) + "\n</svg>\n"


def build(specs: bool) -> str:
    """Build the figure, with the hostnames and the hardware specifications if specs is True."""
    f = Figure()
    llama = "llama-cpp.svg"
    gemma_moe = "Gemma 4 26B A4B QAT"
    # Speeds of single requests with the current configurations, from the READMEs, presets and benchmark/results of
    # the backends. Generation: the generation test of benchmark/llama_cpp_bench_http.py (256 tokens, short prompt)
    # where it is valid, otherwise chat. Prompt processing: a prompt of about 4k tokens where measured, otherwise the
    # nearest length that was measured (in the comments).
    machines = [
        Machine("laptop", 150, pub("ThinkPad L14 Gen 5"), [spec("Core Ultra 5 125U, 32 GB RAM")],
                # llama-cpp-agx-l14/README.md: 26B-A4B without MTP and checkpoints, chat with a 3.5k-token prompt;
                # openvino-agx-l14-npu-e4b/README.md: E4B on the NPU, 684-token prompt
                [(gemma_moe, "8.9", "106"), ("Gemma 4 E4B", "8.1–8.6", "185")], ["kubuntu.svg"],
                [(llama, "llama.cpp", "Vulkan, iGPU"), ("openvino.svg", "OpenVINO", "NPU")]),
        Machine("laptop", 402, pub("ThinkPad T480"), [spec("Core i7-8550U, 32 GB RAM"), pub("GeForce MX150 (2 GB)")],
                # llama-cpp-agx-t480/README.md: 4096-token prompt
                [(gemma_moe, "6.7", "80")], ["kubuntu.svg"], [(llama, "llama.cpp", "CUDA + CPU")]),
        Machine("laptop", 654, spec("Windows laptop"),
                [spec("Core i7-1260P, 32 GB RAM"), pub("T550 Laptop GPU (4 GB)")],
                # llama-cpp-t550/README.md, natively on Windows: 26B-A4B chat (the generation test was not valid),
                # E4B; 4096-token prompts
                [(gemma_moe, "7.9", "92"), ("Gemma 4 E4B", "31", "144")], ["windows.svg"],
                [(llama, "llama.cpp", "CUDA + CPU")]),
        Machine("desktop", 925, spec("agx-z2e"),
                [spec("Threadripper 3970X, 128 GB RAM"), pub("RTX 3090 (24 GB)"), pub("Radeon VII (16 GB)")],
                # llama-cpp-agx-z2e/README.md and llama-cpp-radeon-vii/README.md (image v0.6.0):
                # 6.4k-token prompt (depth test)
                [("Gemma 4 31B QAT", "77", "1060"), (gemma_moe, "124", "1260")], ["kubuntu.svg"],
                [(llama, "llama.cpp", "CUDA, RTX 3090"), (llama, "llama.cpp", "ROCm, Radeon VII")]),
        Machine("server", 1198, spec("agx-ai (agx-h12)"), [spec("EPYC 7302, 256 GB RAM"), pub("RTX 3070 (8 GB)")],
                # llama-cpp-agx-ai/README.md; benchmark/results/agx-ai.jsonl: 36.5k-token prompt (depth test)
                [(gemma_moe, "55", "1547")], ["proxmox.svg", "ubuntu.svg"], [(llama, "llama.cpp", "CUDA + CPU")]),
        Machine("server", 1452, spec("Big Machine"),
                [spec("Threadripper PRO 3975WX"), spec("192 GB RAM"), pub("2 × RTX A4000 (16 GB)")],
                # llama-cpp-big-machine/preset.ini: no single-request prompt processing results
                [("PaperQA2 RAG", "", ""), ("Qwen3.8-27B", "50–67", ""), ("Qwen3-Embedding-8B", "", "")],
                ["ubuntu.svg"], [(llama, "llama.cpp", "CUDA")]),
    ]
    if not specs:
        machines = [m.public() for m in machines]
    text_lines = max(m.text_lines() for m in machines)
    machines_height = Figure.machine_height(text_lines, max(len(m.runtimes) for m in machines))

    # The backends box at the bottom gets the height that the machines need, and the top half gets the rest.
    cw, ch = 230, 62  # Card size
    first_row = 64
    margin = 24  # Between the machines and the bottom of the group boxes
    group_header = 36  # Between the top of a group box and the top of the devices
    hw_header = 56  # Between the top of the backends box and the top of the group boxes
    hw_top = HEIGHT - 24 - margin - machines_height - group_header - hw_header
    row_spacing = min(100, (hw_top - 12 - 18 - ch - first_row) / 3)
    rows = tuple(first_row + i * row_spacing for i in range(4))  # Card rows of the top half

    def mid(y):
        return y + ch / 2

    # Regions
    stack = (196, 12, 904, rows[-1] + ch + 18 - 12)
    f.rect(*stack, fill=STACK_FILL, stroke=STACK_STROKE, rx=16, width=2)
    docker_w = f.logo("docker.svg", stack[0] + 18, stack[1] + 12, 28)
    stack_title = "docker-compose.yml on agx-ai" if specs else "docker-compose.yml"
    f.text(stack[0] + 18 + docker_w + 12, stack[1] + 34, stack_title, size=20, weight="bold")

    internet = (1140, 12, 448, rows[1] + ch + 16 - 12)
    f.rect(*internet, fill=CLOUD_FILL, stroke=CLOUD_STROKE, rx=16, width=2, dash="8 6")
    f.text(internet[0] + 18, internet[1] + 34, "Internet", size=20, weight="bold")

    research = (12, stack[1] + stack[3] - 170, 150, 170)
    f.rect(*research, fill=CLOUD_FILL, stroke=CLOUD_STROKE, rx=16, width=2, dash="8 6")
    f.text(research[0] + 75, research[1] + 25, "Research", size=17, weight="bold", anchor="middle")

    hw_top = stack[1] + stack[3] + 12  # Larger than above if the row spacing was limited
    f.rect(12, hw_top, 1576, HEIGHT - 12 - hw_top, fill=HW_FILL, stroke=HW_STROKE, rx=16, width=2)

    # Users
    f.person_icon(50, 56, 56)
    f.text(78, 136, "Users", size=18, weight="bold", anchor="middle")
    f.text(78, 154, "browser,", size=13, color=MUTED, anchor="middle")
    f.text(78, 170, "Conduit app", size=13, color=MUTED, anchor="middle")

    # Stack
    col_a = 222
    f.card(col_a, rows[0], cw, ch, f.terminal_icon, "Open Terminal", "shell for the LLMs", logo_size=40)
    f.card(col_a, rows[1], cw, ch, "open-webui.png", "Open WebUI", "chat UI", logo_size=40)
    f.card(col_a, rows[2], cw, ch, "gptr.png", "GPT Researcher", "web UI", logo_size=40)
    f.card(col_a, rows[3], cw, ch, "gptr.png", "GPT Researcher", "research agent", logo_size=40)

    lx, lw, lh = 534, 250, 116
    ly = (mid(rows[1]) + mid(rows[2])) / 2 - lh / 2
    f.rect(lx, ly, lw, lh, fill="#ffffff", stroke=STACK_STROKE, width=2.5)
    f.logo("litellm-icon.png", lx + 14, ly + lh / 2 - 34, 68, 68)
    f.text(lx + 94, ly + lh / 2 - 7, "LiteLLM", size=24, weight="bold")
    f.text(lx + 94, ly + lh / 2 + 15, "LLM proxy and router", size=14, color=MUTED)
    f.text(lx + 94, ly + lh / 2 + 33, "OpenAI API", size=14, color=MUTED)

    col_d = 850
    pg_y = (rows[1] + rows[2]) / 2
    f.card(col_d, rows[0], cw, ch, "claude-symbol.svg", "LiteLLM Claude", "subscription → API", logo_size=36)
    f.card(col_d, pg_y, cw, ch, "postgresql.svg", "PostgreSQL", "LiteLLM database", logo_size=40)
    f.card(col_d, rows[3], cw, ch, f.key_icon, "SSH tunnel", "ssh-big-machine" if specs else "to the remote server",
           logo_size=40)

    # Internet
    f.card(1160, rows[0], 410, ch, "claude-symbol.svg", "Claude", "Anthropic, Pro/Max subscription", logo_size=36,
           stroke=CLOUD_STROKE)
    f.card(1160, rows[1], 410, ch, f.key_icon, "SSH jump host",
           f"to the network of the {'Big Machine' if specs else 'remote server'}", logo_size=40, stroke=CLOUD_STROKE)

    # Services used by GPT Researcher
    for i, logo in enumerate(("tavily.svg", "arxiv.svg", "gemini-icon.svg")):
        y = research[1] + 36 + i * 42
        f.rect(research[0] + 12, y, 126, 36, fill="#ffffff", stroke=CLOUD_STROKE, rx=8)
        if logo == "arxiv.svg":
            f.logo(logo, research[0] + 26, y + 6, 24, 98)
        else:
            f.logo(logo, research[0] + 22, y + 6, 24, 24)
            f.text(research[0] + 56, y + 24, "Tavily" if logo == "tavily.svg" else "Gemini", size=17, weight="bold")

    # Arrows in the top half
    users_x = 180
    f.arrow([(106, 90), (users_x, 90), (users_x, mid(rows[1])), (col_a, mid(rows[1]))])
    f.arrow([(users_x, mid(rows[1])), (users_x, mid(rows[2])), (col_a, mid(rows[2]))])
    f.arrow([(col_a + cw / 2, rows[1]), (col_a + cw / 2, rows[0] + ch)])
    f.text(col_a + cw / 2 + 8, rows[1] - 7, "tools", size=13, color=MUTED, style="italic")
    f.arrow([(col_a + cw / 2, rows[2] + ch), (col_a + cw / 2, rows[3])])
    f.arrow([(col_a, mid(rows[3])), (research[0] + research[2], mid(rows[3]))])
    f.arrow([(col_a + cw, mid(rows[1])), (lx, mid(rows[1]))])
    f.arrow([(col_a + cw, mid(rows[3])), (lx - 40, mid(rows[3])), (lx - 40, ly + lh - 20), (lx, ly + lh - 20)])
    f.arrow([(lx + lw - 60, ly), (lx + lw - 60, mid(rows[0])), (col_d, mid(rows[0]))])
    f.arrow([(lx + lw, mid(pg_y)), (col_d, mid(pg_y))])
    f.arrow([(lx + lw, ly + lh - 20), (col_d - 30, ly + lh - 20), (col_d - 30, mid(rows[3])), (col_d, mid(rows[3]))])
    f.arrow([(col_d + cw, mid(rows[0])), (1160, mid(rows[0]))])
    f.arrow([(col_d + cw, mid(rows[3])), (1120, mid(rows[3])), (1120, mid(rows[1])), (1160, mid(rows[1]))])

    # llama.cpp backends
    f.text(34, hw_top + 30, "llama.cpp backends", size=20, weight="bold")
    f.text(240, hw_top + 30, "OpenAI-compatible API in the LAN", size=14, color=MUTED, style="italic")
    # Right of the line from LiteLLM to the backends
    f.text(
        lx + 76, hw_top + 30,
        "t/s of a single request: (generation with a short prompt | prompt processing of a 0.7–37k-token prompt)",
        size=14, color=MUTED, style="italic",
    )

    group_top = hw_top + hw_header
    for name, gx, gw in (("Laptops", 26, 752), ("Desktops", 790, 270), ("Servers", 1072, 504)):
        f.rect(gx, group_top, gw, HEIGHT - 24 - group_top, fill="#ffffff", stroke=HW_STROKE, rx=12, width=1.5,
               extra=' fill-opacity="0.6"')
        f.text(gx + 14, group_top + 24, name, size=16, weight="bold", color="#3d7a2c")

    # Center the machines vertically in the group boxes.
    space = HEIGHT - 24 - group_top - group_header
    top = group_top + group_header + max(0, (space - machines_height) / 2)
    for m in machines:
        f.machine(m, top, text_lines)

    # LiteLLM to the backends in the LAN
    bus_y = hw_top + 43
    litellm_x = lx + 60
    f.add(f'<path d="M{litellm_x},{ly + lh} L{litellm_x},{bus_y} M150,{bus_y} L1198,{bus_y}" '
          f'stroke="{LINE}" stroke-width="2" fill="none"/>')
    f.add(f'<circle cx="{litellm_x}" cy="{bus_y}" r="4" fill="{LINE}"/>')
    for m in machines[:-1]:
        f.arrow([(m.cx, bus_y), (m.cx, top - 4)])
    # The Big Machine is in another network behind the SSH jump host.
    big = machines[-1]
    f.arrow([(big.cx, rows[1] + ch), (big.cx, top - 4)], dashed=True)
    f.text(big.cx + 10, (rows[1] + ch + hw_top) / 2 + 5, "SSH tunnel", size=13, color=MUTED, style="italic")

    return f.render()


def crop_png_height(png: pathlib.Path, height: int) -> None:
    """Crop a non-interlaced PNG to its top height rows.

    Each scanline is filtered only relative to the scanlines above it, so the rows can be cut from the bottom
    without decoding the filters.
    """
    data = png.read_bytes()
    pos = 8
    chunks = []
    while pos < len(data):
        length, = struct.unpack(">I", data[pos:pos + 4])
        chunks.append((data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]))
        pos += 12 + length
    ihdr = chunks[0][1]
    width, old_height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", ihdr)
    if interlace:
        raise ValueError(f"Interlaced PNGs are not supported: {png}")
    if height >= old_height:
        return
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    row_bytes = 1 + (width * channels * bit_depth + 7) // 8
    pixels = zlib.decompress(b"".join(body for kind, body in chunks if kind == b"IDAT"))[:height * row_bytes]

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    out = [data[:8], chunk(b"IHDR", struct.pack(">II", width, height) + ihdr[8:])]
    for kind, body in chunks[1:]:
        if kind == b"IDAT":
            if pixels is not None:
                out.append(chunk(b"IDAT", zlib.compress(pixels, 9)))
                pixels = None
        else:
            out.append(chunk(kind, body))
    png.write_bytes(b"".join(out))


def render_png(svg: pathlib.Path, png: pathlib.Path) -> None:
    """Render the SVG to PNG with headless Chrome.

    Inkscape would drop the space between the <tspan> elements of the runtime tags.
    """
    chrome = next(
        (shutil.which(name) for name in ("google-chrome", "chromium", "chromium-browser") if shutil.which(name)),
        None,
    )
    if chrome is None:
        raise RuntimeError("Chrome or Chromium is required for the PNG output")
    subprocess.run(
        [
            chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
            f"--window-size={WIDTH},{HEIGHT + PNG_EXTRA_HEIGHT}", f"--force-device-scale-factor={PNG_SCALE}",
            f"--screenshot={png}", svg.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    crop_png_height(png, HEIGHT * PNG_SCALE)


if __name__ == "__main__":
    for out, specs in ((OUT, False), (OUT_SPECS, True)):
        out.write_text(build(specs), encoding="utf-8")
        print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KiB)")
        png = out.with_suffix(".png")
        render_png(out, png)
        print(f"Wrote {png} ({png.stat().st_size / 1024:.0f} KiB)")
