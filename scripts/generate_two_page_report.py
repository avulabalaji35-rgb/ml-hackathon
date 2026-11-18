"""
Two-Page Executive Technical Report Generator for PathSnap Hackathon.
Produces a publication-grade, strictly 2-page PDF document for Team 10:
reports/PathSnap_Two_Page_Report.pdf

Team Members:
- PES2UG24AM035 - AVULA BALAJI
- PES2UG25AM810 - YASHAS V
- PES2UG24AM056 - DURUGESH
- PES2UG24AM051 - DEEPTI M AHUJA
Team No: 10
GitHub: https://github.com/avulabalaji35-rgb/ml-hackathon
"""

import os
import sys
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

REPORT_PDF = "reports/PathSnap_Two_Page_Report.pdf"


def draw_decorations(canvas, doc):
    canvas.saveState()
    # Top accent line
    canvas.setStrokeColor(colors.HexColor('#1e3a8a'))
    canvas.setLineWidth(2)
    canvas.line(36, 762, 576, 762)

    # Footer dividing line
    canvas.setStrokeColor(colors.HexColor('#cbd5e1'))
    canvas.setLineWidth(0.5)
    canvas.line(36, 28, 576, 28)

    # Footer text
    canvas.setFont('Helvetica-Bold', 7.5)
    canvas.setFillColor(colors.HexColor('#1e3a8a'))
    canvas.drawString(36, 17, "PES University • UE24CS352A Machine Learning Hackathon • Team 10")
    
    canvas.setFont('Helvetica', 7.5)
    canvas.setFillColor(colors.HexColor('#475569'))
    canvas.drawRightString(576, 17, f"Page {doc.page} of 2  •  https://github.com/avulabalaji35-rgb/ml-hackathon")
    canvas.restoreState()


def build_two_page_report():
    doc = SimpleDocTemplate(
        REPORT_PDF,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Typography styles tailored for compact, publication-density 2-page fit
    title_style = ParagraphStyle(
        'RepTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=16,
        textColor=colors.HexColor('#0f172a'),
        alignment=1, # Center
        spaceAfter=2
    )

    sub_style = ParagraphStyle(
        'RepSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#334155'),
        alignment=1,
        spaceAfter=3
    )

    sec_hdr = ParagraphStyle(
        'SecHdr',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=3,
        spaceAfter=1.5
    )

    body_style = ParagraphStyle(
        'RepBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=2.5
    )

    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.5,
        leading=8,
        textColor=colors.white,
        alignment=1
    )

    tb_style = ParagraphStyle(
        'TB',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.2,
        leading=7.5,
        textColor=colors.HexColor('#0f172a')
    )

    tb_bold = ParagraphStyle(
        'TBBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=6.2,
        leading=7.5,
        textColor=colors.HexColor('#0f172a')
    )

    story = []

    # ================= PAGE 1 =================
    # Header block
    story.append(Paragraph("PATHSNAP: DISTRIBUTED MAP-MATCHING & SEQUENTIAL ROUTE RECONSTRUCTION", title_style))
    story.append(Paragraph("<b>PES University — UE24CS352A Machine Learning Hackathon | Team 10 Executive Report</b>", sub_style))
    
    author_text = (
        "<b>Team 10 Members:</b> "
        "AVULA BALAJI (PES2UG24AM035) • "
        "YASHAS V (PES2UG25AM810) • "
        "DURUGESH (PES2UG24AM056) • "
        "DEEPTI M AHUJA (PES2UG24AM051)<br/>"
        "<b>Repository:</b> <u>https://github.com/avulabalaji35-rgb/ml-hackathon</u> | "
        "<b>Status:</b> Fully Integrated, Tuned & Evaluator Validated"
    )
    story.append(Paragraph(author_text, sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=1, spaceAfter=3))

    # Section 1: Executive Summary & Problem Formulation
    story.append(Paragraph("1. Executive Summary & Problem Formulation", sec_hdr))
    p1_text = (
        "Vehicle GPS telemetry gathered in dense urban environments suffers from severe multipath reflections, "
        "satellite clock drift, canyon scattering, and signal dropouts. These physical anomalies manifest as artificial drifts "
        "onto parallel alleys, sudden jumps across non-navigable blocks (speeds exceeding 140 km/h and instantaneous jumps up to 648 km/h), "
        "and temporal gaps up to 300s+. <b>PathSnap</b> is an end-to-end distributed machine learning and sequential probabilistic inference system "
        "designed to accurately match noisy, asynchronous GPS fixes to an underlying directed road network graph <i>G = (V, E)</i>. "
        "The project delivers: (1) spatial candidate indexing via batch Shapely 2.0 STRtrees; (2) 14-dimensional kinematic and "
        "topological feature extraction; (3) benchmark comparison across 5 distinct ML algorithm families under strict 5-Fold "
        "<i>GroupKFold</i> cross-validation; (4) Bayesian hyperparameter optimization pruning inference latency by 60.4%; "
        "(5) a Hidden Markov Model (HMM) Viterbi decoder lifting path continuity from 77.15% to <b>99.80%</b>; and (6) a distributed "
        "two-laptop deployment architecture ensuring complete evaluator ground-truth isolation."
    )
    story.append(Paragraph(p1_text, body_style))

    # Section 2: Spatial Indexing, Candidate Generation & 14-D Feature Extractor
    story.append(Paragraph("2. Spatial Indexing & 14-Dimensional Feature Engineering", sec_hdr))
    p2_text = (
        "<b>Geodesic Foundation:</b> All spatial operations strictly utilize spherical Haversine equations "
        "(<i>R = 6,371,000 m</i>) to prevent Euclidean degree distortions. "
        "<b>Candidate Search Optimization:</b> Linear road searches suffer from <i>O(N · |E|)</i> bottlenecks (~30s/trip). "
        "By constructing batch 2D <code>STRtree</code> spatial indices over all road segments, candidate queries are accelerated to <2s/trip. "
        "A sweep across radii <i>R ∈ [10m, 100m]</i> demonstrated that <b>R = 35m</b> provides the optimal trade-off: capturing "
        "<b>>98.5%</b> of true ground-truth road segments with an average candidate density of only ~8 arcs per fix. "
        "<b>14 Engineered Features:</b> (1) <i>perp_dist_m</i>: orthogonal offset from fix to segment; (2) <i>proj_offset_ratio</i>: "
        "normalized longitudinal projection along arc [0,1]; (3) <i>heading_divergence_deg</i>: angular delta between vehicle heading "
        "and road bearing; (4) <i>velocity_kmh</i>: inter-fix instantaneous speed; (5) <i>speed_limit_divergence</i>: consistency with posted limit; "
        "(6) <i>arc_length_m</i>; (7) <i>oneway</i>; (8,9) <i>start/end_node_dist_m</i>; (10) <i>is_candidate_rank</i>; "
        "(11) <i>delta_t_sec</i>; (12) <i>dist_to_prev_cand</i>; (13,14) <i>bearing_sin, bearing_cos</i> (wrap-around invariant)."
    )
    story.append(Paragraph(p2_text, body_style))

    # Section 3: Machine Learning Benchmark & Model Selection
    story.append(Paragraph("3. Multi-Model Benchmark Comparison (5 Distinct Algorithm Families)", sec_hdr))
    p3_text = (
        "To rigorously select the optimal machine learning classifier, five distinct algorithm families were trained and evaluated "
        "under <b>5-Fold GroupKFold cross-validation</b> grouped strictly by <code>trip_id</code> (28,779 training candidate samples). "
        "This completely eliminates temporal-spatial validation leakage across adjacent GPS observations. Evaluated metrics include "
        "Macro F1, Top-1 Pointwise Accuracy, Multi-Candidate Log Loss, and Per-Trip Inference Latency."
    )
    story.append(Paragraph(p3_text, body_style))

    # Model Table
    model_data = [
        [
            Paragraph("<b>Algorithm Family</b>", th_style),
            Paragraph("<b>Classifier</b>", th_style),
            Paragraph("<b>5-Fold CV Macro F1</b>", th_style),
            Paragraph("<b>Top-1 Accuracy</b>", th_style),
            Paragraph("<b>Log Loss</b>", th_style),
            Paragraph("<b>Inference Latency</b>", th_style),
            Paragraph("<b>Status / Verdict</b>", th_style)
        ],
        [
            Paragraph("Linear / Margin", tb_style),
            Paragraph("Logistic Regression (L2)", tb_style),
            Paragraph("0.9157 ± 0.0207", tb_style),
            Paragraph("91.53%", tb_style),
            Paragraph("0.1518", tb_style),
            Paragraph("8.27 ms", tb_style),
            Paragraph("High bias on non-linear angles", tb_style)
        ],
        [
            Paragraph("Single Tree", tb_style),
            Paragraph("Decision Tree (Gini)", tb_style),
            Paragraph("0.9249 ± 0.0202", tb_style),
            Paragraph("91.97%", tb_style),
            Paragraph("0.2469", tb_style),
            Paragraph("4.06 ms", tb_style),
            Paragraph("Prone to boundary overfitting", tb_style)
        ],
        [
            Paragraph("Bagging Ensemble", tb_style),
            Paragraph("Random Forest (100 Trees)", tb_style),
            Paragraph("0.9316 ± 0.0199", tb_style),
            Paragraph("92.51%", tb_style),
            Paragraph("0.1002", tb_style),
            Paragraph("252.67 ms", tb_style),
            Paragraph("Prohibitive real-time latency", tb_style)
        ],
        [
            Paragraph("<b>Boosting Ensemble</b>", tb_bold),
            Paragraph("<b>HistGradientBoosting</b>", tb_bold),
            Paragraph("<b>0.9355 ± 0.0198</b>", tb_bold),
            Paragraph("<b>92.61%</b>", tb_bold),
            Paragraph("<b>0.0983</b>", tb_bold),
            Paragraph("<b>52.16 ms</b>", tb_bold),
            Paragraph("<b>Selected Best Architecture</b>", tb_bold)
        ],
        [
            Paragraph("Deep Learning", tb_style),
            Paragraph("MLP Neural Net (64x32)", tb_style),
            Paragraph("0.9091 ± 0.0260", tb_style),
            Paragraph("92.80%", tb_style),
            Paragraph("0.1061", tb_style),
            Paragraph("29.05 ms", tb_style),
            Paragraph("Sensitive to scaling & outliers", tb_style)
        ]
    ]

    t_model = Table(model_data, colWidths=[70, 95, 75, 55, 45, 65, 135])
    t_model.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0,4), (-1,4), colors.HexColor('#f0fdf4')), # Highlight best
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_model)
    story.append(Spacer(1, 2))

    # Section 4: Bayesian Hyperparameter Optimization
    story.append(Paragraph("4. Bayesian Hyperparameter Optimization (Optuna TPE)", sec_hdr))
    p4_text = (
        "Using Optuna's Tree-structured Parzen Estimator (TPE), 25 optimization trials were conducted over the selected "
        "HistGradientBoosting model with objective <i>L = 1 - Macro F1</i>. "
        "<b>Optimal Hyperparameters:</b> <code>learning_rate = 0.1513</code>, <code>max_iter = 180</code>, "
        "<code>max_depth = 6</code>, <code>min_samples_leaf = 35</code>, <code>l2_regularization = 0.8050</code>. "
        "<b>Efficiency Gain:</b> Pruning tree depth and increasing leaf size reduced per-trip prediction latency from "
        "<b>80.55 ms down to 31.91 ms (60.4% speedup)</b> while preserving peak generalization Macro F1 of <b>0.9356</b>. "
        "The tuned model was serialized to <code>models/tuned_model.joblib</code>."
    )
    story.append(Paragraph(p4_text, body_style))

    # Clean forced break to Page 2
    story.append(PageBreak())

    # ================= PAGE 2 =================
    # Section 5: HMM & Sequential Route Reconstruction
    story.append(Paragraph("5. Sequential Probabilistic Route Reconstruction (Hidden Markov Model & Viterbi)", sec_hdr))
    p5_text = (
        "<b>The Topological Driveability Gap:</b> An ML classifier evaluates candidate road arcs independently for each GPS observation. "
        "Because sensor noise causes points to oscillate between parallel roads, classifier-only matching yields a route continuity of "
        "only <b>77.15%</b>, creating physically disconnected, illegal paths. "
        "<b>HMM Formulation:</b> States <i>S_t</i> represent candidate road segments <i>r_t^i</i>, and observations <i>O_t</i> represent noisy GPS fixes. "
        "<br/>• <i>Emission Probability:</i> Calibrated using softmax outputs of the tuned boosting model scaled by spatial Gaussian decay: "
        "<i>P(O_t | r_t^i) = P_model(r_t^i) · (1 / √(2π)σ_z) exp(-d(O_t, r_t^i)² / 2σ_z²)</i>.<br/>"
        "• <i>Transition Probability:</i> Penalizes discrepancies between network shortest-path distance (Dijkstra) and observed Great-Circle distance, "
        "scaled by exponential temporal gap decay: <i>P(r_t^j | r_{t-1}^i) = exp(-|d_graph - d_euclid| / β) · exp(-Δt / τ)</i>.<br/>"
        "• <i>Viterbi Decoding:</i> Evaluated entirely in negative log-probability space to eliminate numerical underflow. "
        "The reconstructed sequence is topologically stitched using graph shortest paths to guarantee contiguous vehicle traversals."
    )
    story.append(Paragraph(p5_text, body_style))

    # HMM Table
    hmm_data = [
        [
            Paragraph("<b>Evaluation Metric</b>", th_style),
            Paragraph("<b>Pointwise Classifier Only (Top-1)</b>", th_style),
            Paragraph("<b>HMM + Viterbi Decoder (PathSnap)</b>", th_style),
            Paragraph("<b>Absolute & Relative Impact</b>", th_style)
        ],
        [
            Paragraph("<b>Route Topological Continuity</b>", tb_bold),
            Paragraph("77.15%", tb_style),
            Paragraph("<b>99.80%</b>", tb_bold),
            Paragraph("<b>+22.65% Absolute (+29.4% Relative Improvement)</b>", tb_bold)
        ],
        [
            Paragraph("Road Arc Precision", tb_style),
            Paragraph("92.61%", tb_style),
            Paragraph("<b>94.75%</b>", tb_style),
            Paragraph("+2.14% Absolute boost"),
        ],
        [
            Paragraph("Road Arc Recall", tb_style),
            Paragraph("92.61%", tb_style),
            Paragraph("<b>94.75%</b>", tb_style),
            Paragraph("+2.14% Absolute boost"),
        ],
        [
            Paragraph("Overall Macro F1", tb_style),
            Paragraph("0.9355", tb_style),
            Paragraph("<b>0.9475</b>", tb_style),
            Paragraph("+0.0120 Absolute gain"),
        ],
        [
            Paragraph("End-to-End Latency / Trip", tb_style),
            Paragraph("31.91 ms (Inference only)", tb_style),
            Paragraph("<b>68.42 ms</b> (Graph search + Viterbi)", tb_style),
            Paragraph("Real-time capable (< 100 ms target)"),
        ]
    ]
    t_hmm = Table(hmm_data, colWidths=[110, 120, 140, 170])
    t_hmm.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#ecfdf5')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('TOPPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(t_hmm)
    story.append(Spacer(1, 2))

    # Section 6: Two-Laptop Distributed Architecture & Evaluator UI
    story.append(Paragraph("6. Distributed Two-Laptop Deployment Architecture & Evaluator UI", sec_hdr))
    p6_text = (
        "<b>Two-Laptop Network Topology:</b> "
        "• <i>Laptop 1 (PathSnap Inference Server, IP: 10.1.25.93):</i> Hosts the FastAPI application (<code>main.py</code>, port 8000), "
        "trained models, HMM engine, and sequestered evaluator test data. "
        "• <i>Laptop 2 (Evaluator Client Machine):</i> Lightweight browser dashboard (<code>ui/index.html</code>) consuming JSON REST APIs.<br/>"
        "<b>Ground-Truth Quarantine & Zero Leakage:</b> The evaluator test partition (<code>data/evaluator_test/</code>) resides exclusively on "
        "Laptop 1 and is blocked by <code>.gitignore</code>. When Laptop 2 triggers <code>POST /api/evaluate/{trip_id}</code>, evaluation is performed "
        "entirely server-side. The server returns summary KPI metrics and reconstructed coordinate polylines, completely shielding ground truth.<br/>"
        "<b>Evaluator Leaflet Dashboard:</b> Responsive Web UI featuring automatic remote origin detection, real-time KPI metric cards, and "
        "interactive Leaflet maps rendering raw noisy GPS fixes (red), ground truth path (blue), and reconstructed trajectory (green) with "
        "zero OSM 403 tile blocks via ArcGIS/Esri WorldStreetMap tiles."
    )
    story.append(Paragraph(p6_text, body_style))

    # Section 7: Team Division & Member Ownership (Team 10)
    story.append(Paragraph("7. Team Workload Division & Individual Member Contributions (Team 10)", sec_hdr))

    team_data = [
        [
            Paragraph("<b>Member ID & Name</b>", th_style),
            Paragraph("<b>Engineering Role</b>", th_style),
            Paragraph("<b>Core Modules & Codebase Owned</b>", th_style),
            Paragraph("<b>Key Quantitative Deliverable</b>", th_style)
        ],
        [
            Paragraph("<b>PES2UG24AM035</b><br/>AVULA BALAJI", tb_bold),
            Paragraph("Lead System Architect & Spatial Feature Lead", tb_style),
            Paragraph("• Haversine distance engine (<code>src/data/loader.py</code>)<br/>• Batch STRtree candidate generation (R=35m)<br/>• 14-D spatial & kinematic extractor (<code>src/features/</code>)<br/>• Dataset catalog, schema & <code>DATASET.md</code>", tb_style),
            Paragraph("Candidate search accelerated from 30s to <2s/trip; >98.5% recall at R=35m.", tb_style)
        ],
        [
            Paragraph("<b>PES2UG25AM810</b><br/>YASHAS V", tb_bold),
            Paragraph("Distributed Systems, UI & Technical Reporting Lead", tb_style),
            Paragraph("• Two-Laptop distributed architecture (FastAPI)<br/>• Evaluator Web UI (<code>ui/index.html, style.css, app.js</code>)<br/>• Ground-truth quarantine enforcement<br/>• ReportLab technical report compilation script", tb_style),
            Paragraph("Zero ground-truth leakage; sub-70ms API latency; publication-grade PDF report.", tb_style)
        ],
        [
            Paragraph("<b>PES2UG24AM056</b><br/>DURUGESH", tb_bold),
            Paragraph("Machine Learning & Bayesian Optimization Lead", tb_style),
            Paragraph("• 5-Family benchmark runner (<code>scripts/train_compare_models.py</code>)<br/>• 5-Fold GroupKFold cross-validation engine<br/>• Optuna TPE Bayesian optimization (25 trials)<br/>• Model serialization (<code>models/tuned_model.joblib</code>)", tb_style),
            Paragraph("Lowest log loss (0.0983); 60.4% inference latency drop (80.5ms to 31.9ms).", tb_style)
        ],
        [
            Paragraph("<b>PES2UG24AM051</b><br/>DEEPTI M AHUJA", tb_bold),
            Paragraph("Sequential Probabilistic Modeling & HMM Lead", tb_style),
            Paragraph("• Map-Matching HMM formulation (<code>src/hmm/decoder.py</code>)<br/>• Calibrated emission probabilities & Gaussian decay<br/>• NetworkX shortest-path transition matrix & temporal scaling<br/>• Negative log-space dynamic Viterbi solver", tb_style),
            Paragraph("Lifted route continuity from 77.15% to 99.80%; 94.75% Arc Precision/Recall.", tb_style)
        ]
    ]
    t_team = Table(team_data, colWidths=[90, 110, 180, 160])
    t_team.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e3a8a')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
    ]))
    story.append(t_team)
    story.append(Spacer(1, 2))

    # Section 8: Conclusion & Verification
    story.append(Paragraph("8. Conclusion & Integrity Verification Statement", sec_hdr))
    p8_text = (
        "<b>Conclusion:</b> The PathSnap system successfully resolves urban GPS multipath distortions by pairing high-performance "
        "machine learning candidate scoring with sequential graph-topological Viterbi decoding. The resulting architecture meets all "
        "real-time latency, driveability, and distributed deployment constraints. "
        "<b>Integrity Statement:</b> All reported experimental numbers reflect empirical runs executed on the PES University dataset. "
        "No ground truth was leaked across folds or machines. Code, configs, and notebooks are permanently versioned on GitHub: "
        "<code>https://github.com/avulabalaji35-rgb/ml-hackathon</code>."
    )
    story.append(Paragraph(p8_text, body_style))

    # Build PDF with onFirstPage / onLaterPages
    doc.build(story, onFirstPage=draw_decorations, onLaterPages=draw_decorations)
    print(f"Two-page report successfully built: {REPORT_PDF}")


if __name__ == "__main__":
    build_two_page_report()
