import json
from io import BytesIO


def _line_items_from_summary(weekly_summary):
    if not weekly_summary:
        return []

    return [
        f"Total check-ins (7d): {weekly_summary['total_checkins']}",
        f"Dominant state: {weekly_summary['dominant_state']}",
        f"Trend direction: {weekly_summary['trend_direction']}",
        f"Average text confidence: {weekly_summary['average_text_confidence'] * 100:.1f}%",
        f"Average urgency: {weekly_summary['average_urgency'] * 100:.1f}%",
        f"Uncertain sessions: {weekly_summary['uncertain_sessions']}",
        f"Highest urgency session: {weekly_summary['highest_urgency_state']} ({weekly_summary['highest_urgency_score'] * 100:.1f}%)",
        f"Signal agreement count: {weekly_summary['agreement_count']}",
        f"Signal disagreement count: {weekly_summary['disagreement_count']}",
        f"Raw label mismatch count: {weekly_summary['raw_label_mismatch_count']}",
    ]


def _line_items_from_diagnostics(diagnostics):
    lines = []
    for row in diagnostics[:5]:
        created_at, mental_state, face_emotion, text_conf, face_conf, text_raw, face_raw, text_top, face_top = row
        text_items = ", ".join(
            f"{item['label']} {item['score'] * 100:.1f}%"
            for item in json.loads(text_top or "[]")
        )
        face_items = ", ".join(
            f"{item['label']} {item['score'] * 100:.1f}%"
            for item in json.loads(face_top or "[]")
        )
        lines.append(f"{created_at} | Final: {mental_state} | Face: {face_emotion}")
        lines.append(f"  Text raw: {text_raw} | Face raw: {face_raw}")
        lines.append(f"  Text top 3: {text_items or 'N/A'}")
        lines.append(f"  Face top 3: {face_items or 'N/A'}")
        lines.append(
            f"  Confidence: text {(text_conf or 0) * 100:.1f}% | face {(face_conf or 0) * 100:.1f}%"
        )
    return lines


def _line_items_from_changes(change_events):
    lines = []
    for event in change_events[:5]:
        lines.append(
            f"{event['date']} | {event['from_state']} -> {event['to_state']} | {event['movement']}"
        )
        lines.append(
            f"  Triggers: {', '.join(event['triggers'])} | "
            f"Urgency delta {event['urgency_delta']:+.2f} | "
            f"Text confidence delta {event['text_conf_delta']:+.2f}"
        )
    return lines


def build_pdf_report(user_name, weekly_summary, diagnostics, change_events):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    def write_line(text, size=11, bold=False, gap=18):
        nonlocal y
        if y < 60:
            pdf.showPage()
            y = height - 50
        font_name = "Helvetica-Bold" if bold else "Helvetica"
        pdf.setFont(font_name, size)
        pdf.drawString(40, y, text[:110])
        y -= gap

    write_line("MindScope AI Clinical-Style Summary Report", size=16, bold=True, gap=24)
    write_line(f"User: {user_name}", bold=True)
    write_line("Purpose: AI-assisted emotional check-in summary for reflection and review.")
    write_line("Disclaimer: This document is not a medical diagnosis and should not replace professional care.", gap=24)

    write_line("Weekly Summary", size=13, bold=True, gap=20)
    for line in _line_items_from_summary(weekly_summary):
        write_line(f"- {line}")

    write_line("Model Diagnostics Snapshot", size=13, bold=True, gap=20)
    for line in _line_items_from_diagnostics(diagnostics):
        write_line(line)

    write_line("Change Detection", size=13, bold=True, gap=20)
    for line in _line_items_from_changes(change_events or []):
        write_line(line)

    write_line("Interpretation Notes", size=13, bold=True, gap=20)
    write_line("- Use this report to review trends, uncertainty, and repeated model disagreement.")
    write_line("- High urgency or high-alert sessions should be treated as prompts for human follow-up.")
    write_line("- Low-confidence or uncertain outputs should not be over-interpreted.")

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()
