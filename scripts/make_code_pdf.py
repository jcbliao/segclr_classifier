"""Typeset this project's source tree into one readable, navigable PDF.

Output: gnn/code_pdf/gnn_classifier_source.pdf -- a title page, a table of
contents with page numbers, PDF bookmarks (one per file, grouped by section),
and every file syntax-highlighted with line numbers in a gutter.

Only this project's own code is included; the vendored clones (segclr_db/,
segCLR_cell_classification/), caches, results and logs are not.

Two render passes: pass 1 draws into a throwaway buffer purely to learn which
page each file starts on, pass 2 draws for real with those numbers in the TOC.
Pagination does not depend on page numbers, so the two passes agree -- the
front-matter page count is known up front from the file list alone.

Run via scripts/sbatch/make_code_pdf.sh (never on a login node).
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pygments import lex
from pygments.lexers import BashLexer, MarkdownLexer, PythonLexer, TextLexer
from pygments.token import Token
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdfcanvas

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- what to include

#: (section title, [paths relative to the repo root]). Order is reading order:
#: the project's own docs, then the model, then the data layer, then entry points.
SECTIONS: list[tuple[str, list[str]]] = [
    (
        "Documentation",
        ["CLAUDE.md", "README.md", "data/DENDRITE_THICKNESS.md", "data/DEPRECATED.md"],
    ),
    (
        "gnn/ -- the model",
        [
            "gnn/model.py",
            "gnn/graph_transformer.py",
            "gnn/encoder.py",
            "gnn/readout.py",
            "gnn/lcpn.py",
            "gnn/resnet.py",
            "gnn/hierarchy.py",
            "gnn/metrics.py",
            "gnn/__init__.py",
        ],
    ),
    (
        "data/ -- the data layer",
        [
            "data/build_dataset_from_store.py",
            "data/build_window_membership.py",
            "data/geodesic_window.py",
            "data/dataset_windowed.py",
            "data/dataset_lcpn.py",
            "data/dendrite_thickness.py",
            "data/neuron_mesh.py",
            "data/build_dendrite_thickness.py",
            "data/build_dataset.py",
            "data/dataset.py",
            "data/public_reader.py",
            "data/cave_skeletons.py",
            "data/__init__.py",
        ],
    ),
    ("scripts/ -- entry points", ["scripts/*.py"]),
    ("scripts/sbatch/ -- batch wrappers", ["scripts/sbatch/*.sh"]),
]

# ---------------------------------------------------------------- page geometry

PAGE_W, PAGE_H = letter
M_LEFT, M_RIGHT, M_TOP, M_BOTTOM = 52, 40, 52, 46

FONT, FONT_BOLD, FONT_ITALIC = "Mono", "Mono-Bold", "Mono-Italic"
CODE_SIZE = 8.0
CODE_LEAD = 10.0
GUTTER = 26.0  # line-number column, right-aligned into it

HEADER_Y = PAGE_H - M_TOP + 20
FOOTER_Y = M_BOTTOM - 22
BODY_TOP = PAGE_H - M_TOP
BODY_BOTTOM = M_BOTTOM

TEXT_X = M_LEFT + GUTTER
TEXT_W = PAGE_W - M_RIGHT - TEXT_X

# ---------------------------------------------------------------- colors

INK = HexColor("#1b1f24")
FAINT = HexColor("#9aa4ae")
RULE = HexColor("#d7dce1")
ACCENT = HexColor("#1f6feb")

#: Longest-prefix wins, so Token.Name.Function beats Token.Name. Each entry is
#: (color, bold, italic).
TOKEN_STYLE: list[tuple[object, tuple[str, bool, bool]]] = [
    (Token.Comment, ("#6a737d", False, True)),
    (Token.Keyword, ("#cf222e", False, False)),
    (Token.Keyword.Constant, ("#0550ae", False, False)),
    (Token.Keyword.Type, ("#0550ae", False, False)),
    (Token.Name.Builtin, ("#0550ae", False, False)),
    (Token.Name.Function, ("#6639ba", True, False)),
    (Token.Name.Class, ("#6639ba", True, False)),
    (Token.Name.Decorator, ("#9a6700", False, False)),
    (Token.Name.Namespace, ("#6639ba", False, False)),
    (Token.Name.Exception, ("#6639ba", True, False)),
    (Token.Name.Variable, ("#1b1f24", False, False)),
    (Token.String, ("#0a6640", False, False)),
    # Not italic: module docstrings here run to 60+ lines, and a page of italic
    # monospace is markedly harder to read than the short comments above.
    (Token.String.Doc, ("#0a6640", False, False)),
    (Token.String.Escape, ("#9a6700", False, False)),
    (Token.String.Interpol, ("#9a6700", False, False)),
    (Token.Number, ("#0550ae", False, False)),
    (Token.Operator, ("#5b6672", False, False)),
    (Token.Punctuation, ("#5b6672", False, False)),
    (Token.Generic.Heading, ("#6639ba", True, False)),
    (Token.Generic.Subheading, ("#6639ba", True, False)),
    (Token.Generic.Strong, ("#1b1f24", True, False)),
    (Token.Generic.Emph, ("#1b1f24", False, True)),
    (Token.Error, ("#cf222e", False, False)),
]
DEFAULT_STYLE = ("#1b1f24", False, False)

_style_cache: dict[object, tuple[HexColor, str]] = {}


def style_for(tok) -> tuple[HexColor, str]:
    """Resolve a pygments token to (color, font name), walking up its parents."""
    if tok in _style_cache:
        return _style_cache[tok]
    best: tuple[str, bool, bool] = DEFAULT_STYLE
    best_depth = -1
    for prefix, style in TOKEN_STYLE:
        if tok in prefix and len(prefix) > best_depth:
            best, best_depth = style, len(prefix)
    color, bold, italic = best
    font = FONT_BOLD if bold else (FONT_ITALIC if italic else FONT)
    _style_cache[tok] = (HexColor(color), font)
    return _style_cache[tok]


# ---------------------------------------------------------------- source loading


@dataclass
class SourceFile:
    section: str
    path: str  # repo-relative
    lines: list[list[tuple[str, str]]]  # per line: [(text, token-name)] runs
    n_lines: int


def lexer_for(path: Path):
    match path.suffix:
        case ".py":
            return PythonLexer(stripnl=False)
        case ".sh":
            return BashLexer(stripnl=False)
        case ".md":
            return MarkdownLexer(stripnl=False)
        case _:
            return TextLexer(stripnl=False)


def tokenize(path: Path) -> list[list[tuple[str, object]]]:
    """Lex a file into a list of lines, each a list of (text, token) runs."""
    text = path.read_text(encoding="utf-8", errors="replace").expandtabs(4)
    lines: list[list[tuple[str, object]]] = [[]]
    for tok, value in lex(text, lexer_for(path)):
        # A single pygments run can straddle newlines (docstrings, comments).
        parts = value.split("\n")
        for i, part in enumerate(parts):
            if i:
                lines.append([])
            if part:
                lines[-1].append((part, tok))
    while lines and not lines[-1]:
        lines.pop()
    return lines


def collect() -> list[SourceFile]:
    files: list[SourceFile] = []
    for section, patterns in SECTIONS:
        paths: list[Path] = []
        for pattern in patterns:
            if "*" in pattern:
                parent, _, glob = pattern.rpartition("/")
                paths.extend(sorted((REPO / parent).glob(glob)))
            else:
                p = REPO / pattern
                if p.exists():
                    paths.append(p)
                else:
                    print(f"  ! missing, skipped: {pattern}")
        for p in paths:
            rel = str(p.relative_to(REPO))
            lines = tokenize(p)
            files.append(SourceFile(section, rel, lines, len(lines)))
    return files


# ---------------------------------------------------------------- the renderer


class Book:
    """Draws the document. Set `dry` to paginate without producing real output."""

    def __init__(
        self,
        canvas: pdfcanvas.Canvas,
        front_pages: int,
        dry: bool,
        known_starts: dict[str, int] | None = None,
    ):
        self.c = canvas
        self.dry = dry
        self.front_pages = front_pages
        # Pass 1 has nothing to cite yet, so its contents page prints zeros; pass 2
        # gets pass 1's map. Either way the entry count -- and so the pagination --
        # is identical, which is what makes the two passes agree.
        self.known_starts = known_starts or {}
        self.page = 1
        self.y = BODY_TOP
        self.header = ""
        self.char_w = pdfmetrics.stringWidth("x" * 100, FONT, CODE_SIZE) / 100.0
        self.max_cols = max(20, int(TEXT_W / self.char_w))
        self.starts: dict[str, int] = {}

    # -- page furniture ----------------------------------------------------

    def _furniture(self) -> None:
        c = self.c
        if self.header:
            c.setFont(FONT, 7.0)
            c.setFillColor(FAINT)
            c.drawString(M_LEFT, HEADER_Y, self.header)
            c.drawRightString(PAGE_W - M_RIGHT, HEADER_Y, "gnn_classifier")
            c.setStrokeColor(RULE)
            c.setLineWidth(0.4)
            c.line(M_LEFT, HEADER_Y - 5, PAGE_W - M_RIGHT, HEADER_Y - 5)
        c.setFont(FONT, 7.5)
        c.setFillColor(FAINT)
        c.drawCentredString(PAGE_W / 2, FOOTER_Y, str(self.page))

    def new_page(self) -> None:
        self._furniture()
        self.c.showPage()
        self.page += 1
        self.y = BODY_TOP

    def need(self, height: float) -> None:
        if self.y - height < BODY_BOTTOM:
            self.new_page()

    # -- front matter ------------------------------------------------------

    def title_page(self, files: list[SourceFile], commit: str) -> None:
        c = self.c
        total_lines = sum(f.n_lines for f in files)
        y = PAGE_H * 0.62

        c.setFillColor(ACCENT)
        c.rect(M_LEFT, y + 46, 62, 3, stroke=0, fill=1)

        c.setFillColor(INK)
        c.setFont(FONT_BOLD, 25)
        c.drawString(M_LEFT, y, "gnn_classifier")
        c.setFont(FONT, 11.5)
        c.setFillColor(HexColor("#5b6672"))
        y -= 22
        c.drawString(M_LEFT, y, "Complete source listing")
        y -= 40
        c.setFont(FONT, 8.6)
        c.setFillColor(INK)
        for label, value in [
            ("files", str(len(files))),
            ("lines", f"{total_lines:,}"),
            ("commit", commit),
            ("generated", dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
        ]:
            c.setFillColor(FAINT)
            c.drawString(M_LEFT, y, f"{label:<11}")
            c.setFillColor(INK)
            c.drawString(M_LEFT + 62, y, value)
            y -= 14

        y -= 20
        c.setFillColor(HexColor("#5b6672"))
        c.setFont(FONT, 8.0)
        for line in [
            "SegCLR cell-type classification -- per-window GNN aggregation over",
            "geodesic context windows, LCPN hierarchical head.",
            "",
            "Vendored clones (segclr_db/, segCLR_cell_classification/) are excluded.",
        ]:
            c.drawString(M_LEFT, y, line)
            y -= 11.5

        self._furniture()
        c.showPage()
        self.page += 1
        self.y = BODY_TOP

    def toc(self, files: list[SourceFile]) -> None:
        """Render the table of contents. Consumes exactly `front_pages - 1` pages."""
        c = self.c
        start_page = self.page
        self.header = "Contents"
        y = BODY_TOP - 6
        c.setFillColor(INK)
        c.setFont(FONT_BOLD, 14)
        c.drawString(M_LEFT, y, "Contents")
        y -= 26

        section = None
        for f in files:
            if y < BODY_BOTTOM + 20:
                self._furniture()
                c.showPage()
                self.page += 1
                y = BODY_TOP - 6
            if f.section != section:
                section = f.section
                y -= 6
                c.setFillColor(ACCENT)
                c.setFont(FONT_BOLD, 9)
                c.drawString(M_LEFT, y, section)
                y -= 14
            page = self.known_starts.get(f.path, 0)
            c.setFillColor(INK)
            c.setFont(FONT, 8.2)
            c.drawString(M_LEFT + 12, y, f.path)
            label_w = pdfmetrics.stringWidth(f.path, FONT, 8.2)
            num = str(page)
            num_w = pdfmetrics.stringWidth(num, FONT, 8.2)
            c.setFillColor(RULE)
            dot_x0 = M_LEFT + 18 + label_w
            dot_x1 = PAGE_W - M_RIGHT - num_w - 6
            if dot_x1 > dot_x0:
                c.setLineWidth(0.4)
                c.setDash(0.6, 3)
                c.line(dot_x0, y + 2.4, dot_x1, y + 2.4)
                c.setDash()
            c.setFillColor(FAINT)
            c.drawRightString(PAGE_W - M_RIGHT, y, num)
            y -= 12.4

        self._furniture()
        c.showPage()
        self.page += 1
        self.y = BODY_TOP
        used = self.page - start_page
        expected = self.front_pages - 1
        if used != expected:
            # Pagination is deterministic, so this only fires if toc_page_count()
            # and this loop disagree -- fail loudly rather than ship wrong numbers.
            raise RuntimeError(f"TOC used {used} pages, front matter reserved {expected}")

    # -- body --------------------------------------------------------------

    def file_heading(self, f: SourceFile) -> None:
        c = self.c
        self.header = f.path
        y = self.y  # always BODY_TOP: a file only ever opens on a fresh page
        # Sits a full title-height below BODY_TOP, or it rides up into the
        # running header band that _furniture() draws above the margin.
        c.setFillColor(ACCENT)
        c.rect(M_LEFT, y - 14, 3, 13, stroke=0, fill=1)
        c.setFillColor(INK)
        c.setFont(FONT_BOLD, 12.5)
        c.drawString(M_LEFT + 11, y - 12, f.path)
        c.setFillColor(FAINT)
        c.setFont(FONT, 7.5)
        c.drawRightString(PAGE_W - M_RIGHT, y - 12, f"{f.n_lines} lines")
        c.setStrokeColor(RULE)
        c.setLineWidth(0.6)
        c.line(M_LEFT, y - 24, PAGE_W - M_RIGHT, y - 24)
        self.y = y - 38

    def wrap(self, runs: list[tuple[str, object]]) -> list[list[tuple[str, object]]]:
        """Split one source line's runs into visual lines of at most max_cols chars.

        Continuation lines keep the leading indent of the original so wrapped code
        stays visually attached to its own line rather than to the left margin.
        """
        flat = "".join(text for text, _ in runs)
        if len(flat) <= self.max_cols:
            return [runs]
        indent = len(flat) - len(flat.lstrip(" "))
        hang = min(indent + 4, max(0, self.max_cols - 20))
        out: list[list[tuple[str, object]]] = []
        col: int = 0
        cur: list[tuple[str, object]] = []
        for text, tok in runs:
            while text:
                room = self.max_cols - col
                if room <= 0:
                    out.append(cur)
                    cur, col = [(" " * hang, Token.Text)], hang
                    continue
                chunk, text = text[:room], text[room:]
                cur.append((chunk, tok))
                col += len(chunk)
        if cur:
            out.append(cur)
        return out

    def draw_line(self, lineno: int, runs: list[tuple[str, object]], first: bool) -> None:
        c = self.c
        self.need(CODE_LEAD)
        y = self.y - CODE_SIZE
        if first:
            c.setFont(FONT, 6.4)
            c.setFillColor(FAINT)
            c.drawRightString(M_LEFT + GUTTER - 8, y, str(lineno))
        t = c.beginText(TEXT_X, y)
        for text, tok in runs:
            if not text.strip():
                t.setFont(FONT, CODE_SIZE)
                t.textOut(text)
                continue
            color, font = style_for(tok)
            t.setFillColor(color)
            t.setFont(font, CODE_SIZE)
            t.textOut(text)
        c.drawText(t)
        self.y -= CODE_LEAD

    def add_file(self, f: SourceFile) -> None:
        self.starts[f.path] = self.page
        if not self.dry:
            key = f"file:{f.path}"
            self.c.bookmarkPage(key)
            self.c.addOutlineEntry(f.path, key, level=1)
        self.file_heading(f)
        for i, runs in enumerate(f.lines, start=1):
            for j, visual in enumerate(self.wrap(runs)):
                self.draw_line(i, visual, first=(j == 0))

    def finish(self) -> None:
        """Flush the page in progress. No page break follows, so no blank trailer."""
        self._furniture()
        self.c.showPage()


def toc_page_count(files: list[SourceFile]) -> int:
    """Pages the contents listing will take -- must match Book.toc()'s own loop."""
    y = BODY_TOP - 6 - 26
    pages, section = 1, None
    for f in files:
        if y < BODY_BOTTOM + 20:
            pages += 1
            y = BODY_TOP - 6
        if f.section != section:
            section = f.section
            y -= 20
        y -= 12.4
    return pages


def render(
    files: list[SourceFile],
    out: object,
    front_pages: int,
    dry: bool,
    commit: str,
    known_starts: dict[str, int] | None = None,
) -> Book:
    c = pdfcanvas.Canvas(out, pagesize=letter, invariant=True)
    c.setTitle("gnn_classifier -- complete source listing")
    c.setAuthor("jcbliao")
    c.setSubject("SegCLR GNN cell-type classifier")
    book = Book(c, front_pages, dry, known_starts)
    book.title_page(files, commit)
    book.toc(files)
    section = None
    for i, f in enumerate(files):
        if i:
            book.new_page()  # every file opens on a fresh page
        if not dry and f.section != section:
            section = f.section
            key = f"sec:{f.section}"
            c.bookmarkPage(key)
            c.addOutlineEntry(f.section, key, level=0, closed=False)
        book.add_file(f)
    book.finish()
    c.save()
    return book


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "gnn/code_pdf/gnn_classifier_source.pdf"))
    args = ap.parse_args()

    fonts = (
        REPO / "segclr_db/.venv/lib/python3.11/site-packages/matplotlib/mpl-data/fonts/ttf"
    )
    pdfmetrics.registerFont(TTFont(FONT, str(fonts / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, str(fonts / "DejaVuSansMono-Bold.ttf")))
    pdfmetrics.registerFont(TTFont(FONT_ITALIC, str(fonts / "DejaVuSansMono-Oblique.ttf")))

    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    print("collecting sources...")
    files = collect()
    print(f"  {len(files)} files, {sum(f.n_lines for f in files):,} lines")

    front_pages = 1 + toc_page_count(files)
    print(f"front matter: {front_pages} pages")

    print("pass 1: paginating...")
    dry = render(files, io.BytesIO(), front_pages, dry=True, commit=commit)
    print(f"  {dry.page} pages")

    print("pass 2: rendering...")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    final = render(files, str(out), front_pages, dry=False, commit=commit, known_starts=dry.starts)

    if final.starts != dry.starts:
        raise RuntimeError("page numbering differed between passes")
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} -- {final.page} pages, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
