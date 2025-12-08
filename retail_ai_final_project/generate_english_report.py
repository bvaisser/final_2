#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate English PDF Report - Coffee Sales Analysis
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime
import os

# Path to artifacts directory
ARTIFACTS_DIR = "artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")

def create_english_pdf():
    """Create English PDF Report"""

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(ARTIFACTS_DIR, f"english_coffee_analysis_{timestamp}.pdf")

    # Create PDF document
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Elements list
    elements = []

    # Styles
    styles = getSampleStyleSheet()

    # Main title style
    title_style = ParagraphStyle(
        'EnglishTitle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1f4788')
    )

    # Heading style
    heading_style = ParagraphStyle(
        'EnglishHeading',
        parent=styles['Heading2'],
        alignment=TA_LEFT,
        fontSize=18,
        spaceAfter=12,
        textColor=colors.HexColor('#2c5aa0')
    )

    # Normal text style
    normal_style = ParagraphStyle(
        'EnglishNormal',
        parent=styles['Normal'],
        alignment=TA_LEFT,
        fontSize=11,
        spaceAfter=12,
        leading=16
    )

    # Bullet style
    bullet_style = ParagraphStyle(
        'EnglishBullet',
        parent=styles['Normal'],
        alignment=TA_LEFT,
        fontSize=11,
        spaceAfter=8,
        leading=16,
        leftIndent=20
    )

    # Main title
    elements.append(Paragraph("Coffee Sales Analysis - Insights and Conclusions", title_style))
    elements.append(Spacer(1, 0.3*inch))

    # Date
    date_text = f"Report Date: {datetime.now().strftime('%m/%d/%Y')}"
    elements.append(Paragraph(date_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))

    # Performance Summary
    elements.append(Paragraph("Model Performance Summary", heading_style))
    elements.append(Paragraph(
        "The developed model achieved an exceptional accuracy of <b>99.89%</b> (R² = 0.9989) in predicting coffee sales. "
        "The model processed <b>3,636 records</b> and created <b>19 engineered features</b> for prediction.",
        normal_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    # Performance metrics
    performance_data = [
        ['Metric', 'Value'],
        ['R² Score', '0.9989'],
        ['RMSE', '0.0760'],
        ['MAE', '0.0426'],
        ['MSE', '0.0058']
    ]

    perf_table = Table(performance_data, colWidths=[2*inch, 2*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(perf_table)
    elements.append(Spacer(1, 0.4*inch))

    # Coffee Consumption Times - Hourly
    elements.append(Paragraph("🕐 Coffee Consumption Times - Hourly Analysis", heading_style))

    elements.append(Paragraph("<b>Peak Consumption Times:</b>", normal_style))
    elements.append(Paragraph("• <b>10:00 AM</b> - Absolute peak (349 sales) - Classic morning coffee time", bullet_style))
    elements.append(Paragraph("• <b>11:00 AM</b> - Second peak (294 sales) - Continuation of morning trend", bullet_style))
    elements.append(Paragraph("• <b>4:00 PM (16:00)</b> - Third peak (282 sales) - Late afternoon break", bullet_style))
    elements.append(Paragraph("• <b>10:00 PM (22:00)</b> - Fourth peak (~310 sales) - Surprising evening consumption", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("<b>Temporal Insights:</b>", normal_style))
    elements.append(Paragraph("• There are <b>3 clear peaks</b>: Morning (8-12), Afternoon (15-17), and Late evening (19-22)", bullet_style))
    elements.append(Paragraph("• Hours 8-12 are the most concentrated peak hours", bullet_style))
    elements.append(Paragraph("• Significant drop during 13-14 (lunch hours)", bullet_style))
    elements.append(Paragraph("• Renewed increase from afternoon through evening", bullet_style))
    elements.append(Spacer(1, 0.3*inch))

    # Add hour distribution chart
    hour_dist_img = os.path.join(FIGURES_DIR, "hour_of_day_distribution.png")
    if os.path.exists(hour_dist_img):
        img = Image(hour_dist_img, width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))

    # Distribution by time of day
    elements.append(Paragraph("📅 Distribution by Time of Day", heading_style))

    time_data = [
        ['Time of Day', 'Number of Sales', 'Percentage'],
        ['Afternoon', '1,231', '33.9%'],
        ['Morning', '1,221', '33.6%'],
        ['Night', '1,184', '32.6%']
    ]

    time_table = Table(time_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    time_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(time_table)
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(
        "<b>Conclusion:</b> Almost perfectly uniform distribution across the three time periods - no clear dominance!",
        normal_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    # Weekly patterns
    elements.append(Paragraph("📆 Weekly Patterns", heading_style))
    elements.append(Paragraph("<b>Busiest Days:</b>", normal_style))
    elements.append(Paragraph("• <b>Tuesday</b> - Busiest day (~580 sales)", bullet_style))
    elements.append(Paragraph("• <b>Friday</b> - Second busiest (~540 sales)", bullet_style))
    elements.append(Paragraph("• <b>Sunday</b> - Quietest day (~440 sales)", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(
        "<b>Insight:</b> Mid-week days (Tuesday-Friday) are busier than weekends, "
        "indicating a working/student customer base.",
        normal_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    # Add weekday chart
    weekday_img = os.path.join(FIGURES_DIR, "Weekdaysort_distribution.png")
    if os.path.exists(weekday_img):
        img = Image(weekday_img, width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))

    # New page
    elements.append(PageBreak())

    # Monthly patterns
    elements.append(Paragraph("📊 Monthly/Seasonal Patterns", heading_style))
    elements.append(Paragraph("<b>Busiest Months:</b>", normal_style))
    elements.append(Paragraph("• <b>March</b> (Month 3) - Absolute peak (~520 sales)", bullet_style))
    elements.append(Paragraph("• <b>October</b> (Month 10) - High (~420 sales)", bullet_style))
    elements.append(Paragraph("• <b>May-June</b> - Quietest months", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(
        "<b>Insight:</b> Clear seasonality - Spring and Fall are busier, Summer is quieter (possibly due to vacations).",
        normal_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    # Add month chart
    month_img = os.path.join(FIGURES_DIR, "Monthsort_distribution.png")
    if os.path.exists(month_img):
        img = Image(month_img, width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))

    # Coffee types
    elements.append(Paragraph("☕ Preferred Coffee Types", heading_style))

    coffee_data = [
        ['Rank', 'Coffee Type', 'Number of Sales', 'Percentage'],
        ['🥇', 'Americano with Milk', '824', '22.7%'],
        ['🥈', 'Latte', '782', '21.5%'],
        ['🥉', 'Americano', '578', '15.9%'],
        ['4', 'Cappuccino', '501', '13.8%'],
        ['5', 'Cortado', '292', '8.0%'],
        ['6', 'Hot Chocolate', '282', '7.8%'],
        ['7', 'Cocoa', '243', '6.7%'],
        ['8', 'Espresso', '134', '3.7%']
    ]

    coffee_table = Table(coffee_data, colWidths=[0.7*inch, 2*inch, 1.3*inch, 1*inch])
    coffee_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8B4513')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (0, 3), colors.HexColor('#F5DEB3')),
        ('BACKGROUND', (0, 4), (-1, -1), colors.HexColor('#FFF8DC')),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(coffee_table)
    elements.append(Spacer(1, 0.4*inch))

    # Business insights
    elements.append(Paragraph("💡 Key Business Insights", heading_style))

    elements.append(Paragraph("<b>1. Milk-Based Coffee Dominance:</b>", normal_style))
    elements.append(Paragraph(
        "• 72% of sales are milk-based beverages (Americano with Milk + Latte + Cappuccino)",
        bullet_style
    ))
    elements.append(Paragraph("• Customers prefer softer drinks over black coffee", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("<b>2. Time Distribution:</b>", normal_style))
    elements.append(Paragraph(
        "• The model identified <b>hour_of_day as the most important feature (30.55%)</b> for predicting sales",
        bullet_style
    ))
    elements.append(Paragraph("• <b>Implication:</b> Time of day is the most critical factor for sales success", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("<b>3. Operational Recommendations:</b>", normal_style))
    elements.append(Paragraph("• <b>Strengthen staffing</b> during 10-11 AM and 4:00 PM", bullet_style))
    elements.append(Paragraph("• Prepare <b>larger inventory</b> of Americano with Milk and Latte", bullet_style))
    elements.append(Paragraph("• <b>Tuesday-Friday</b> require more staff and inventory", bullet_style))
    elements.append(Paragraph("• <b>Reduce inventory</b> during summer months (May-June)", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("<b>4. Surprises:</b>", normal_style))
    elements.append(Paragraph(
        "• <b>10:00 PM</b> is a significant peak time - consider extending opening hours",
        bullet_style
    ))
    elements.append(Paragraph(
        "• Uniform distribution across morning/afternoon/evening indicates <b>diverse customer base</b>",
        bullet_style
    ))
    elements.append(Spacer(1, 0.4*inch))

    # Feature importance
    elements.append(Paragraph("🎯 Feature Importance in the Model", heading_style))

    feature_data = [
        ['Feature', 'Importance'],
        ['hour_of_day (Hour of day)', '30.55%'],
        ['hour_of_day_binned (Binned hour)', '25.47%'],
        ['hour_of_day_squared (Hour squared)', '22.20%'],
        ['Weekdaysort_squared (Weekday squared)', '3.77%'],
        ['Weekdaysort (Day of week)', '3.53%'],
        ['Time_of_Day (Time period)', '3.28%'],
        ['Monthsort (Month)', '2.60%']
    ]

    feature_table = Table(feature_data, colWidths=[3.5*inch, 1.5*inch])
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(feature_table)
    elements.append(Spacer(1, 0.3*inch))

    # Add feature importance chart
    feature_img = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    if os.path.exists(feature_img):
        elements.append(PageBreak())
        elements.append(Paragraph("Feature Importance Chart", heading_style))
        img = Image(feature_img, width=6*inch, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))

    # Summary
    elements.append(PageBreak())
    elements.append(Paragraph("📋 Summary and Conclusions", heading_style))
    elements.append(Paragraph(
        "The coffee sales analysis revealed clear patterns that enable business operational optimization:",
        normal_style
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(
        "✓ <b>Peak Times:</b> Three main peaks - Morning (10-11 AM), Afternoon (4 PM), and Evening (10 PM)",
        bullet_style
    ))
    elements.append(Paragraph(
        "✓ <b>Product Preferences:</b> 72% of customers prefer coffee with milk",
        bullet_style
    ))
    elements.append(Paragraph(
        "✓ <b>Weekly Patterns:</b> Mid-week days are busier than weekends",
        bullet_style
    ))
    elements.append(Paragraph(
        "✓ <b>Seasonality:</b> Spring and Fall are busy, Summer is quieter",
        bullet_style
    ))
    elements.append(Paragraph(
        "✓ <b>Model Accuracy:</b> 99.89% - Highly reliable prediction for future sales",
        bullet_style
    ))
    elements.append(Spacer(1, 0.4*inch))

    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        alignment=TA_CENTER,
        fontSize=9,
        textColor=colors.grey
    )
    elements.append(Paragraph(
        f"This report was automatically generated using CrewAI | {datetime.now().strftime('%m/%d/%Y %H:%M')}",
        footer_style
    ))

    # Build PDF
    doc.build(elements)

    return output_file

if __name__ == "__main__":
    try:
        pdf_file = create_english_pdf()
        print(f"✅ English PDF report created successfully: {pdf_file}")
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        import traceback
        traceback.print_exc()
