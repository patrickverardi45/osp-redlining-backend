from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle
from reportlab.pdfgen import canvas


def _fmt(value: Any, default: str = "-") -> str:
    if value is None or value == "":
        return default
    return str(value)


def _escape(text: Any) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_escape(text), style)


def _draw_header(c: canvas.Canvas, title: str, subtitle: str) -> None:
    width, height = letter

    c.setFillColor(colors.HexColor("#0f172a"))
    c.rect(0, height - 1.0 * inch, width, 1.0 * inch, fill=1, stroke=0)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(0.65 * inch, height - 0.58 * inch, title)

    c.setFont("Helvetica", 10)
    c.drawString(0.65 * inch, height - 0.82 * inch, subtitle)


def _draw_summary_cards(c: canvas.Canvas, summary: Dict[str, Any], route_station: str, route_ft: float) -> float:
    width, height = letter
    top_y = height - 1.45 * inch

    card_w = 1.45 * inch
    card_h = 0.8 * inch
    gap = 0.10 * inch
    start_x = 0.52 * inch

    cards = [
        ("Route Length", f"{route_station} ({route_ft:,.0f} ft)"),
        ("Points", _fmt(summary.get("total_points", 0))),
        ("High Risk", _fmt(summary.get("high_risk", 0))),
        ("Shallow", _fmt(summary.get("shallow_depth", 0))),
        ("Depth Jump", _fmt(summary.get("depth_jump", 0))),
        ("Station Gap", _fmt(summary.get("station_gap", 0))),
        ("BOC Jump", _fmt(summary.get("boc_jump", 0))),
    ]

    for i, (label, value) in enumerate(cards):
        x = start_x + i * (card_w + gap)

        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.roundRect(x, top_y - card_h, card_w, card_h, 8, fill=1, stroke=1)

        c.setFillColor(colors.HexColor("#475569"))
        c.setFont("Helvetica", 8)
        c.drawString(x + 0.10 * inch, top_y - 0.23 * inch, label)

        c.setFillColor(colors.HexColor("#0f172a"))
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(x + 0.10 * inch, top_y - 0.48 * inch, value)

    return top_y - 1.0 * inch


def _draw_coverage_bar(c: canvas.Canvas, coverage: Dict[str, Any]) -> float:
    width, height = letter
    y = height - 2.65 * inch
    x = 0.65 * inch
    bar_w = 6.8 * inch
    bar_h = 0.18 * inch

    covered_pct = float(coverage.get("coverage_pct_of_route", 0.0) or 0.0)
    covered_w = max(0.0, min(bar_w, bar_w * (covered_pct / 100.0)))

    c.setFillColor(colors.HexColor("#475569"))
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y + 0.28 * inch, "Coverage Summary")

    c.setFillColor(colors.HexColor("#e2e8f0"))
    c.roundRect(x, y, bar_w, bar_h, 4, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#0f766e"))
    c.roundRect(x, y, covered_w, bar_h, 4, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#334155"))
    c.setFont("Helvetica", 8)
    c.drawString(
        x,
        y - 0.18 * inch,
        f"Covered: {_fmt(coverage.get('covered_station'))} ({_fmt(coverage.get('covered_ft'))} ft)   "
        f"Coverage: {_fmt(coverage.get('coverage_pct_of_route'))}%"
    )

    return y - 0.35 * inch


def _status_color(flags: List[str], offset_color: str):
    if "HIGH_RISK" in flags:
        return colors.HexColor("#dc2626")
    if any(flag in flags for flag in ["SHALLOW_DEPTH", "DEPTH_JUMP", "STATION_GAP", "BOC_JUMP"]):
        return colors.HexColor("#ea580c")
    if "WRONG_STREET" in flags or "OFF_ROUTE" in flags:
        return colors.HexColor("#ea580c")
    if offset_color == "yellow":
        return colors.HexColor("#d97706")
    if offset_color == "green":
        return colors.HexColor("#16a34a")
    return colors.HexColor("#334155")


def _build_detail_table(results: List[Dict[str, Any]]) -> Table:
    body_style = ParagraphStyle(
        "BodyCell",
        fontName="Helvetica",
        fontSize=7,
        leading=8,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=0,
        spaceAfter=0,
    )

    header_style = ParagraphStyle(
        "HeaderCell",
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=8,
        textColor=colors.white,
        spaceBefore=0,
        spaceAfter=0,
    )

    rows = [[
        _p("Seq", header_style),
        _p("Point ID", header_style),
        _p("Station", header_style),
        _p("Depth", header_style),
        _p("BOC", header_style),
        _p("Flags", header_style),
        _p("Confidence", header_style),
        _p("Notes", header_style),
    ]]

    for row in results:
        flags_text = ", ".join(row.get("flags", [])) if row.get("flags") else "OK"

        notes = []
        if row.get("qa_notes"):
            notes.extend(row.get("qa_notes", []))
        if row.get("station_gap_ft"):
            notes.append(f"Gap {row.get('station_gap_ft')} ft")
        if row.get("depth_jump_ft"):
            notes.append(f"Depth Δ {row.get('depth_jump_ft')} ft")
        if row.get("boc_jump"):
            notes.append(f"BOC Δ {row.get('boc_jump')}")

        rows.append([
            _p(_fmt(row.get("sequence")), body_style),
            _p(_fmt(row.get("id")), body_style),
            _p(_fmt(row.get("station")), body_style),
            _p(_fmt(row.get("depth_ft")), body_style),
            _p(_fmt(row.get("rod")), body_style),
            _p(flags_text, body_style),
            _p(_fmt(row.get("confidence")), body_style),
            _p("; ".join(notes) if notes else "-", body_style),
        ])

    table = Table(
        rows,
        colWidths=[
            0.38 * inch,
            0.65 * inch,
            0.55 * inch,
            0.50 * inch,
            0.45 * inch,
            1.70 * inch,
            0.55 * inch,
            2.55 * inch,
        ],
        repeatRows=1,
    )

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#0f172a")),
    ])

    for row_index, row in enumerate(results, start=1):
        text_color = _status_color(row.get("flags", []), row.get("offset_color", "red"))
        style.add("TEXTCOLOR", (5, row_index), (5, row_index), text_color)

    table.setStyle(style)
    return table


def _draw_footer(c: canvas.Canvas, page_num: int) -> None:
    width, _ = letter
    c.setStrokeColor(colors.HexColor("#cbd5e1"))
    c.line(0.65 * inch, 0.55 * inch, width - 0.65 * inch, 0.55 * inch)

    c.setFillColor(colors.HexColor("#64748b"))
    c.setFont("Helvetica", 8)
    c.drawString(0.65 * inch, 0.35 * inch, "OSP Redlining QA Report")
    c.drawRightString(width - 0.65 * inch, 0.35 * inch, f"Page {page_num}")


def build_client_ready_pdf(report_data: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=3.2 * inch,
        bottomMargin=0.8 * inch,
    )

    results = report_data.get("results", [])
    summary = report_data.get("summary", {})
    coverage = report_data.get("coverage", {})
    route_length_ft = float(report_data.get("route_length_ft", 0.0))
    route_length_station = str(report_data.get("route_length_station", "0+00"))

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    subtitle = f"Generated {generated_at}"

    table = _build_detail_table(results)

    def first_page(c: canvas.Canvas, _doc) -> None:
        _draw_header(c, "Fiber Construction QA / OSP Redlining Report", subtitle)
        _draw_summary_cards(c, summary, route_length_station, route_length_ft)
        _draw_coverage_bar(c, coverage)
        _draw_footer(c, 1)

    def later_pages(c: canvas.Canvas, _doc) -> None:
        _draw_header(c, "Fiber Construction QA / OSP Redlining Report", subtitle)
        _draw_footer(c, c.getPageNumber())

    doc.build([table], onFirstPage=first_page, onLaterPages=later_pages)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes