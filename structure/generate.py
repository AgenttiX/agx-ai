#!/usr/bin/env python3
"""Generate agx-ai-structure.svg, a figure of the local AI server setup of ../docker-compose.yml.

The logos in ./logos are embedded as data URIs, so that the figure is a single self-contained file
that works also in an <img> tag (e.g. in reveal.js). Run with: python3 generate.py
"""

import base64
import pathlib
import re
import struct
from dataclasses import dataclass
from xml.sax.saxutils import escape

DIR = pathlib.Path(__file__).resolve().parent
LOGOS = DIR / "logos"
OUT = DIR / "agx-ai-structure.svg"

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

# The text of these logos is small compared to their height, so draw them larger than the others.
LOGO_SCALE = {"ubuntu.svg": 1.7}


def _svg_data(path: pathlib.Path) -> tuple[bytes, float]:
    text = path.read_text(encoding="utf-8")
    root = re.search(r"<svg\b[^>]*>", text, re.S).group(0)
    view_box = re.search(r'viewBox="([^"]+)"', root)
    if view_box:
        _, _, w, h = (float(v) for v in view_box.group(1).replace(",", " ").split())
    else:
        # Without a viewBox, an SVG image is not scaled to the size of the <image> element.
        w = float(re.search(r'\swidth="([\d.]+)', root).group(1))
        h = float(re.search(r'\sheight="([\d.]+)', root).group(1))
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
    kind: str  # laptop, desktop or server
    cx: float  # Center x coordinate
    name: str
    lines: list[str]  # Hardware
    models: list[str]
    os: list[str]  # Logos of the operating system and the virtualization
    hardware: list[str]  # Logos of the hardware vendors
    runtimes: list[tuple[str, str, str]]  # (logo, title, detail) of what runs on the machine


class Figure:
    def __init__(self):
        self.parts: list[str] = []
        self.logos: dict[str, tuple[str, float]] = {}

    def add(self, s: str):
        self.parts.append(s)

    # Primitives

    def _logo_id(self, name: str) -> str:
        return "logo-" + name.rsplit(".", 1)[0]

    def logo_width(self, name: str, h: float) -> float:
        if name not in self.logos:
            self.logos[name] = load_logo(name)
        return h * self.logos[name][1]

    def logo(self, name: str, x: float, y: float, h: float, w: float | None = None):
        """Draw a logo centered in the box (x, y, w, h) and return w. By default, w is the natural width at the height h.

        Each logo is embedded only once as a <symbol>, which is then referenced with <use>.
        """
        if w is None:
            w = self.logo_width(name, h)
        else:
            self.logo_width(name, h)
        self.add(f'<use href="#{self._logo_id(name)}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}"/>')
        return w

    def symbols(self) -> str:
        out = []
        for name, (uri, aspect) in self.logos.items():
            w = 100 * aspect
            out.append(
                f'<symbol id="{self._logo_id(name)}" viewBox="0 0 {w:g} 100">'
                f'<image width="{w:g}" height="100" href="{uri}"/></symbol>'
            )
        return "\n".join(out)

    def text(self, x, y, s, size=14, weight="normal", color=INK, anchor="start", style=""):
        self.add(
            f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}"{f" font-style={chr(34)}{style}{chr(34)}" if style else ""}>{escape(s)}</text>'
        )

    def rect(self, x, y, w, h, fill="#ffffff", stroke=LINE, rx=10, width=1.5, dash=None, extra=""):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{width}"{dash_attr}{extra}/>'
        )

    def arrow(self, points, dashed=False):
        d = "M" + " L".join(f"{x:g},{y:g}" for x, y in points)
        dash_attr = ' stroke-dasharray="7 5"' if dashed else ""
        self.add(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="2"{dash_attr} marker-end="url(#arrow)"/>')

    # Components

    def card(self, x, y, w, h, logo, title, subtitle=None, logo_size=44, fill="#ffffff", stroke=STACK_STROKE):
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

    def terminal_icon(self, x, y, s):
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

    def key_icon(self, x, y, s):
        """OpenSSH has no official logo, so draw a key."""
        c = "#1f2933"
        r = s * 0.2
        cx, cy = x + s * 0.3, y + s * 0.5
        self.add(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" fill="none" stroke="{c}" stroke-width="4"/>')
        self.add(
            f'<path d="M{cx + r:g},{cy:g} L{x + s * 0.92:g},{cy:g} M{x + s * 0.8:g},{cy:g} L{x + s * 0.8:g},{cy + s * 0.17:g} '
            f'M{x + s * 0.68:g},{cy:g} L{x + s * 0.68:g},{cy + s * 0.13:g}" fill="none" stroke="{c}" '
            f'stroke-width="4" stroke-linecap="round"/>'
        )

    def person_icon(self, x, y, s, color=MUTED):
        self.add(f'<circle cx="{x + s / 2:g}" cy="{y + s * 0.28:g}" r="{s * 0.2:g}" fill="{color}"/>')
        self.add(
            f'<path d="M{x + s * 0.1:g},{y + s * 0.95:g} C{x + s * 0.1:g},{y + s * 0.55:g} {x + s * 0.9:g},'
            f'{y + s * 0.55:g} {x + s * 0.9:g},{y + s * 0.95:g} Z" fill="{color}"/>'
        )

    # Devices. (cx, top) is the top center of the drawing, which is 150 px wide and 110 px tall.

    def laptop(self, cx, top):
        self.rect(cx - 68, top + 6, 136, 88, fill="#3e4c59", stroke="#1f2933", rx=7, width=2)
        self.rect(cx - 60, top + 13, 120, 74, fill="#dbe7f3", stroke="none", rx=2, width=0)
        self.add(
            f'<path d="M{cx - 82:g},{top + 97:g} L{cx + 82:g},{top + 97:g} L{cx + 76:g},{top + 108:g} '
            f'L{cx - 76:g},{top + 108:g} Z" fill="#9aa5b1" stroke="#1f2933" stroke-width="2" stroke-linejoin="round"/>'
        )
        self.rect(cx - 16, top + 97, 32, 4, fill="#7b8794", stroke="none", rx=2, width=0)
        self.logo("llama-cpp.svg", cx - 34, top + 16, 68, 68)

    def desktop(self, cx, top):
        # Monitor
        mx = cx - 40
        self.rect(mx - 72, top, 144, 94, fill="#3e4c59", stroke="#1f2933", rx=7, width=2)
        self.rect(mx - 64, top + 8, 128, 76, fill="#dbe7f3", stroke="none", rx=2, width=0)
        self.add(
            f'<path d="M{mx - 10:g},{top + 94:g} L{mx + 10:g},{top + 94:g} L{mx + 14:g},{top + 108:g} '
            f'L{mx - 14:g},{top + 108:g} Z" fill="#9aa5b1" stroke="#1f2933" stroke-width="2" stroke-linejoin="round"/>'
        )
        self.rect(mx - 36, top + 106, 72, 5, fill="#9aa5b1", stroke="#1f2933", rx=2, width=2)
        self.logo("llama-cpp.svg", mx - 34, top + 12, 68, 68)
        # Tower
        tx = cx + 50
        self.rect(tx, top + 6, 50, 105, fill="#3e4c59", stroke="#1f2933", rx=5, width=2)
        for i in range(3):
            self.rect(tx + 8, top + 16 + i * 10, 34, 5, fill="#7b8794", stroke="none", rx=1.5, width=0)
        self.add(f'<circle cx="{tx + 25:g}" cy="{top + 90:g}" r="5" fill="none" stroke="#9aa5b1" stroke-width="2"/>')

    def server(self, cx, top):
        # A rack with three server units, the middle one with a front display showing llama.cpp.
        self.rect(cx - 80, top, 160, 111, fill="#1f2933", stroke="#1f2933", rx=5, width=2)
        for i, y in enumerate((top + 6, top + 40, top + 74)):
            self.rect(cx - 74, y, 148, 31, fill="#3e4c59", stroke="#52606d", rx=3, width=1)
            for j in range(5):
                self.rect(cx - 66 + j * 9, y + 7, 4, 17, fill="#616e7c", stroke="none", rx=1, width=0)
            self.add(f'<circle cx="{cx + 62:g}" cy="{y + 10:g}" r="3.2" fill="#3ebd93"/>')
            self.add(f'<circle cx="{cx + 62:g}" cy="{y + 21:g}" r="3.2" fill="{LLAMA}"/>')
        # Front display
        self.rect(cx - 12, top + 2, 58, 107, fill="#dbe7f3", stroke="#9aa5b1", rx=4, width=1.5)
        self.logo("llama-cpp.svg", cx - 8, top + 30, 50, 50)

    def runtime_tag(self, cx, y, logo, title, detail):
        """A pill that shows what runs on a device, e.g. llama.cpp with CUDA.

        A wordmark logo replaces the title. The text width is estimated for Arial-like fonts,
        as an SVG in an <img> tag can use only the fonts installed on the system.
        """
        wordmark = self.logo_width(logo, 1) > 1.5
        logo_w = self.logo_width(logo, 18) if wordmark else 22
        text_w = 0.55 * 14 * (len(detail) + (0 if wordmark else len(title) + 1))
        w = 6 + logo_w + 8 + text_w + 14
        x = cx - w / 2
        self.rect(x, y, w, 30, fill="#ffffff", stroke=LLAMA if logo == "llama-cpp.svg" else "#6b4fbb", rx=15, width=1.5)
        if wordmark:
            self.logo(logo, x + 10, y + 6, 18, logo_w)
            self.text(x + 10 + logo_w + 8, y + 20, detail, size=14, color=MUTED)
        else:
            self.logo(logo, x + 6, y + 4, 22, 22)
            self.add(
                f'<text x="{x + 34:g}" y="{y + 20:g}" font-size="14" fill="{INK}">'
                f'<tspan font-weight="bold">{escape(title)}</tspan> <tspan fill="{MUTED}">{escape(detail)}</tspan></text>'
            )

    def vendor_logos(self, cx, y, names, h=18, gap=14):
        """Draw a centered row of logos with the height h, or h * LOGO_SCALE for the logos with small text."""
        heights = [h * LOGO_SCALE.get(n, 1) for n in names]
        widths = [self.logo_width(n, lh) for n, lh in zip(names, heights)]
        total = sum(widths) + gap * (len(names) - 1)
        x = cx - total / 2
        for n, w, lh in zip(names, widths, heights):
            self.logo(n, x, y + (h - lh) / 2, lh, w)
            x += w + gap

    def machine(self, m: "Machine", top: float, text_lines: int):
        """Draw a machine. The logo rows and the runtime tags are aligned between the machines,
        so the text block has room for text_lines lines below the name."""
        getattr(self, m.kind)(m.cx, top)
        y = top + 134
        self.text(m.cx, y, m.name, size=18, weight="bold", anchor="middle")
        for line in m.lines:
            y += 18
            self.text(m.cx, y, line, size=14, color=MUTED, anchor="middle")
        for model in m.models:
            y += 18
            self.text(m.cx, y, model, size=14, weight="bold", color=MODEL, anchor="middle")
        y = top + 134 + 18 * text_lines + 18
        self.vendor_logos(m.cx, y, m.os)
        y += 32
        self.vendor_logos(m.cx, y, m.hardware)
        y += 36
        for logo, title, detail in m.runtimes:
            self.runtime_tag(m.cx, y, logo, title, detail)
            y += 35

    def render(self) -> str:
        head = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" font-family="{FONT}">
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


def build() -> str:
    f = Figure()
    cw, ch = 230, 60  # Card size
    rows = (58, 140, 222, 304)  # Card rows of the top half

    def mid(y):
        return y + ch / 2

    # Regions
    stack = (196, 12, 904, rows[-1] + ch + 16 - 12)
    f.rect(*stack, fill=STACK_FILL, stroke=STACK_STROKE, rx=16, width=2)
    docker_w = f.logo("docker.svg", stack[0] + 18, stack[1] + 12, 28)
    f.text(stack[0] + 18 + docker_w + 12, stack[1] + 34, "docker-compose.yml on agx-ai", size=20, weight="bold")

    internet = (1140, 12, 448, rows[1] + ch + 16 - 12)
    f.rect(*internet, fill=CLOUD_FILL, stroke=CLOUD_STROKE, rx=16, width=2, dash="8 6")
    f.text(internet[0] + 18, internet[1] + 34, "Internet", size=20, weight="bold")

    research = (12, 204, 150, stack[1] + stack[3] - 204)
    f.rect(*research, fill=CLOUD_FILL, stroke=CLOUD_STROKE, rx=16, width=2, dash="8 6")
    f.text(research[0] + 75, research[1] + 25, "Research", size=17, weight="bold", anchor="middle")

    hw_top = stack[1] + stack[3] + 12
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

    lx, ly, lw, lh = 534, 146, 250, 110
    f.rect(lx, ly, lw, lh, fill="#ffffff", stroke=STACK_STROKE, width=2.5)
    f.logo("litellm-icon.png", lx + 14, ly + 21, 68, 68)
    f.text(lx + 94, ly + 48, "LiteLLM", size=24, weight="bold")
    f.text(lx + 94, ly + 70, "LLM proxy and router", size=14, color=MUTED)
    f.text(lx + 94, ly + 88, "OpenAI API", size=14, color=MUTED)

    col_d = 850
    pg_y = (rows[1] + rows[2]) / 2
    f.card(col_d, rows[0], cw, ch, "claude-symbol.svg", "LiteLLM Claude", "subscription → API", logo_size=36)
    f.card(col_d, pg_y, cw, ch, "postgresql.svg", "PostgreSQL", "LiteLLM database", logo_size=40)
    f.card(col_d, rows[3], cw, ch, f.key_icon, "SSH tunnel", "ssh-big-machine", logo_size=40)

    # Internet
    f.card(1160, rows[0], 410, ch, "claude-symbol.svg", "Claude", "Anthropic, Pro/Max subscription", logo_size=36,
           stroke=CLOUD_STROKE)
    f.card(1160, rows[1], 410, ch, f.key_icon, "SSH jump host", "to the network of the Big Machine", logo_size=40,
           stroke=CLOUD_STROKE)

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

    group_top = hw_top + 56
    for name, gx, gw in (("Laptops", 26, 752), ("Desktops", 790, 270), ("Servers", 1072, 504)):
        f.rect(gx, group_top, gw, HEIGHT - 24 - group_top, fill="#ffffff", stroke=HW_STROKE, rx=12, width=1.5,
               extra=' fill-opacity="0.6"')
        f.text(gx + 14, group_top + 24, name, size=16, weight="bold", color="#3d7a2c")

    top = group_top + 36
    llama = "llama-cpp.svg"
    gemma_moe = "Gemma 4 26B A4B QAT"
    machines = [
        Machine("laptop", 150, "agx-l14", ["ThinkPad L14 Gen 5", "Core Ultra 5 125U, 32 GB RAM"], [gemma_moe],
                ["kubuntu.svg"], ["intel.svg"],
                [(llama, "llama.cpp", "Vulkan, iGPU"), ("openvino.svg", "OpenVINO", "NPU")]),
        Machine("laptop", 402, "agx-t480", ["ThinkPad T480", "Core i7-8550U, 32 GB RAM", "GeForce MX150 (2 GB)"],
                [gemma_moe], ["kubuntu.svg"], ["intel.svg", "nvidia.svg"], [(llama, "llama.cpp", "CPU + CUDA")]),
        Machine("laptop", 654, "Windows laptop", ["Docker Desktop, WSL 2", "Core i7-1260P, 32 GB RAM",
                                                  "T550 Laptop GPU (4 GB)"],
                [gemma_moe], ["windows.svg"], ["intel.svg", "nvidia.svg"], [(llama, "llama.cpp", "CUDA")]),
        Machine("desktop", 925, "agx-z2e", ["Threadripper 3970X, 128 GB RAM", "RTX 3090 (24 GB)", "Radeon VII (16 GB)"],
                ["Gemma 4 31B QAT", gemma_moe], ["kubuntu.svg"], ["amd.svg", "nvidia.svg"],
                [(llama, "llama.cpp", "CUDA, RTX 3090"), (llama, "llama.cpp", "ROCm, Radeon VII")]),
        Machine("server", 1198, "agx-ai (agx-h12)", ["EPYC 7302, 256 GB RAM", "RTX 3070 (8 GB)"], [gemma_moe],
                ["ubuntu.svg", "proxmox.svg"], ["amd.svg", "nvidia.svg"], [(llama, "llama.cpp", "CUDA + CPU")]),
        Machine("server", 1452, "Big Machine", ["Threadripper PRO 3975WX", "192 GB RAM", "2 × RTX A4000 (16 GB)"],
                ["PaperQA2 RAG"], ["ubuntu.svg"], ["amd.svg", "nvidia.svg"], [(llama, "llama.cpp", "CUDA")]),
    ]
    text_lines = max(len(m.lines) + len(m.models) for m in machines)
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

if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB)")
