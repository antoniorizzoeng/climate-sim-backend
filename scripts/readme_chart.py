"""Draw the README latency comparison from docs/performance.md."""

from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


def text(x, y, value, color="#eeedf1", size=17, align="start"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{align}" font-size="{size}" '
        f'fill="{color}">{escape(value)}</text>'
    )


def main():
    report = (ROOT / "docs/performance.md").read_text()
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 293" '
        'font-family="DejaVu Sans, sans-serif" role="img">',
        "<title>Frame response time before and after renderer changes</title>",
        '<rect width="840" height="293" rx="12" fill="#18191f"/>',
        text(28, 39, "How long does a frame request take?", size=25),
        text(
            28,
            68,
            "4096 × 4096 source  ·  median of nine warm requests  ·  milliseconds",
            "#c4c0ca",
            16,
        ),
        text(625, 108, "Before", "#c2b1e5", 16, "end"),
        text(774, 108, "After", "#ffad86", 16, "end"),
    ]
    rows = [
        line
        for line in report.splitlines()
        if line.startswith(("| Native", "| PNG", "| Float32"))
    ]
    for idx, row in enumerate(rows):
        name, before, after, *_ = [x.strip(" `") for x in row.strip("|").split("|")]
        before, after = [float(t.split()[0]) for t in (before, after)]
        y = 141 + idx * 40
        svg.extend(
            [
                text(28, y, name.replace("`", "")),
                text(625, y, f"{before:,.2f}", "#c2b1e5", align="end"),
                text(774, y, f"{after:,.2f}", "#ffad86", align="end"),
            ]
        )
        for offset, color, value in [(-9, "#c2b1e5", before), (0, "#ffad86", after)]:
            svg.append(
                f'<rect x="335" y="{y+offset}" width="{value/8:.2f}" height="5" '
                f'rx="2" fill="{color}"/>'
            )
    svg.append("</svg>")
    (ROOT / "docs/assets/frame-latency.svg").write_text("\n".join(svg) + "\n")


if __name__ == "__main__":
    main()
