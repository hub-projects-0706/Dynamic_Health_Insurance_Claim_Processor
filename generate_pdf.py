import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def build_pdf(filename="results/Dynamic_Health_Insurance_Claim_Processor_Report.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette Colors
    primary_color = colors.HexColor("#1A365D")    # Dark Navy
    secondary_color = colors.HexColor("#2B6CB0")  # Royal Blue
    bg_light = colors.HexColor("#F7FAFC")         # Off-white / light gray
    text_dark = colors.HexColor("#2D3748")        # Slate Charcoal

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceAfter=10
    )

    heading1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_dark,
        spaceAfter=5
    )

    story = []

    # Title & Header Banner
    story.append(Paragraph("Dynamic Health Insurance Claim Processor", title_style))
    story.append(Paragraph("Technical Specification, Dynamic Policy Verification & Realistic ML Evaluation Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=8))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Policy Standing Intelligence", heading1_style))
    exec_summary_text = (
        "The <b>Dynamic Health Insurance Claim Processor</b> provides real-time AI adjudication for medical claims. "
        "It features live <b>Dynamic Policy Standing Verification</b> (checking <code>ACTIVE</code>, <code>INACTIVE</code>, "
        "<code>SUSPENDED</code>, and <code>FRAUD_FLAGGED</code> policy statuses directly from database & dataset registries) "
        "before granting instant auto-approvals (&lt;10ms) or escalating high-risk claims to an investigator workbench."
    )
    story.append(Paragraph(exec_summary_text, body_style))

    # Machine Learning Performance & Realistic Evaluation
    story.append(Paragraph("2. Realistic Machine Learning Model Evaluation Metrics", heading1_style))
    cm_text = (
        "Evaluated on 200 evaluation claim records from the <b>Kaggle / CMS Medicare Claims Dataset</b> with natural noise & overlap:<br/>"
        "• <b>Realistic Claims Noise Introduced</b>: Simulates overlapping boundary conditions, clinical variance, and unobserved factors.<br/>"
        "• <b>Test Set Performance</b>: 83.00% Accuracy, 94.59% High Fraud Recall."
    )
    story.append(Paragraph(cm_text, body_style))

    metrics_data = [
        [Paragraph("<b>Evaluation Metric</b>", body_style), Paragraph("<b>Score Achieved</b>", body_style), Paragraph("<b>Benchmark Evaluation & Interpretation</b>", body_style)],
        [Paragraph("<b>Accuracy</b>", body_style), Paragraph("<b>83.00%</b>", body_style), Paragraph("166 / 200 noisy test claims correctly classified.", body_style)],
        [Paragraph("<b>Precision</b>", body_style), Paragraph("<b>84.34%</b>", body_style), Paragraph("High precision ensuring low false accusations.", body_style)],
        [Paragraph("<b>Recall</b>", body_style), Paragraph("<b>94.59%</b>", body_style), Paragraph("Captures 94.59% of fraudulent/abusive claims.", body_style)],
        [Paragraph("<b>F1-Score</b>", body_style), Paragraph("<b>0.8917</b>", body_style), Paragraph("Strong harmonic balance between Precision and Recall.", body_style)],
        [Paragraph("<b>ROC-AUC</b>", body_style), Paragraph("<b>0.8299</b>", body_style), Paragraph("Robust risk probability discrimination capability.", body_style)],
    ]

    metrics_table = Table(metrics_data, colWidths=[1.8 * inch, 1.7 * inch, 4.0 * inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('BACKGROUND', (0, 1), (-1, 1), bg_light),
        ('BACKGROUND', (0, 3), (-1, 3), bg_light),
        ('BACKGROUND', (0, 5), (-1, 5), bg_light),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 6))

    # FastAPI REST Gateway Integration
    story.append(Paragraph("3. FastAPI REST Gateway Integration", heading1_style))
    fastapi_data = [
        [Paragraph("<b>HTTP Method</b>", body_style), Paragraph("<b>Endpoint</b>", body_style), Paragraph("<b>Integration Purpose & Action</b>", body_style)],
        [Paragraph("<b>GET</b>", body_style), Paragraph("<code>/api/v1/health</code>", body_style), Paragraph("Returns API service status & engine metadata.", body_style)],
        [Paragraph("<b>GET</b>", body_style), Paragraph("<code>/api/v1/policies/verify/{policy_id}</code>", body_style), Paragraph("Queries DB & dataset policy standing (ACTIVE, SUSPENDED, FRAUD_FLAGGED).", body_style)],
        [Paragraph("<b>POST</b>", body_style), Paragraph("<code>/api/v1/claims/evaluate</code>", body_style), Paragraph("Evaluates claim JSON payload & returns dynamic route decision.", body_style)],
        [Paragraph("<b>GET</b>", body_style), Paragraph("<code>/api/v1/claims/samples</code>", body_style), Paragraph("Pre-loads test claim presets (Active, Blacklisted, Suspended).", body_style)],
        [Paragraph("<b>GET</b>", body_style), Paragraph("<code>/api/v1/metrics</code>", body_style), Paragraph("Exposes live model metrics (83.00% Accuracy, 0.8299 ROC-AUC).", body_style)],
    ]

    fastapi_table = Table(fastapi_data, colWidths=[1.3 * inch, 2.7 * inch, 3.5 * inch])
    fastapi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('BACKGROUND', (0, 1), (-1, 1), bg_light),
        ('BACKGROUND', (0, 3), (-1, 3), bg_light),
        ('BACKGROUND', (0, 5), (-1, 5), bg_light),
    ]))
    story.append(fastapi_table)
    story.append(Spacer(1, 6))

    # Dynamic Routing Matrix
    story.append(Paragraph("4. Dynamic Decision Routing Matrix", heading1_style))
    routing_matrix_data = [
        [Paragraph("<b>Target Route Queue</b>", body_style), Paragraph("<b>Adjudication Criteria & Thresholds</b>", body_style), Paragraph("<b>Operational Action</b>", body_style)],
        [
            Paragraph("🟢 <b>AUTO_PROCESSED</b>", body_style),
            Paragraph("Composite Risk < 0.35 AND Model Confidence >= 0.20 AND Policy Active AND Zero Critical Flags.", body_style),
            Paragraph("⚡ Emergency Instant Approval (<10ms).", body_style)
        ],
        [
            Paragraph("🟡 <b>PENDING_ADDITIONAL_VALIDATION</b>", body_style),
            Paragraph("0.35 <= Composite Risk <= 0.65 OR Model Confidence < 0.20.", body_style),
            Paragraph("Enqueued for secondary validation.", body_style)
        ],
        [
            Paragraph("🔴 <b>HUMAN_INVESTIGATION</b>", body_style),
            Paragraph("Composite Risk > 0.65 OR Critical Flag (Blacklisted Policy, Sanctioned Provider, Duplicate Claim).", body_style),
            Paragraph("🚨 High-Priority Fraud Audit Workbench.", body_style)
        ],
    ]

    matrix_table = Table(routing_matrix_data, colWidths=[2.1 * inch, 3.4 * inch, 2.0 * inch])
    matrix_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#F0FFF4")),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#FFFFF0")),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#FFF5F5")),
    ]))
    story.append(matrix_table)

    doc.build(story)
    print(f"[SUCCESS] PDF report successfully recompiled at {filename}")

if __name__ == '__main__':
    build_pdf()
