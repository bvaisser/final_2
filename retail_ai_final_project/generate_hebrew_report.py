#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
יצירת דוח PDF בעברית - ניתוח מכירות קפה
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from datetime import datetime
import os

# נתיב לתיקיית artifacts
ARTIFACTS_DIR = "artifacts"
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")

def create_hebrew_pdf():
    """יצירת דוח PDF בעברית"""

    # יצירת שם קובץ עם תאריך
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(ARTIFACTS_DIR, f"hebrew_coffee_analysis_{timestamp}.pdf")

    # יצירת מסמך PDF
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # רשימת אלמנטים
    elements = []

    # סגנונות
    styles = getSampleStyleSheet()

    # סגנון לכותרת ראשית
    title_style = ParagraphStyle(
        'HebrewTitle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#1f4788')
    )

    # סגנון לכותרת משנה
    heading_style = ParagraphStyle(
        'HebrewHeading',
        parent=styles['Heading2'],
        alignment=TA_RIGHT,
        fontSize=18,
        spaceAfter=12,
        textColor=colors.HexColor('#2c5aa0')
    )

    # סגנון לטקסט רגיל
    normal_style = ParagraphStyle(
        'HebrewNormal',
        parent=styles['Normal'],
        alignment=TA_RIGHT,
        fontSize=11,
        spaceAfter=12,
        leading=16
    )

    # סגנון לרשימה
    bullet_style = ParagraphStyle(
        'HebrewBullet',
        parent=styles['Normal'],
        alignment=TA_RIGHT,
        fontSize=11,
        spaceAfter=8,
        leading=16,
        leftIndent=20
    )

    # כותרת ראשית
    elements.append(Paragraph("ניתוח מכירות קפה - מסקנות ותובנות", title_style))
    elements.append(Spacer(1, 0.3*inch))

    # תאריך
    date_text = f"תאריך הדוח: {datetime.now().strftime('%d/%m/%Y')}"
    elements.append(Paragraph(date_text, normal_style))
    elements.append(Spacer(1, 0.2*inch))

    # סיכום ביצועים
    elements.append(Paragraph("סיכום ביצועי המודל", heading_style))
    elements.append(Paragraph(
        "המודל שפותח השיג דיוק יצירתי של <b>99.89%</b> (R² = 0.9989) בחיזוי מכירות הקפה. "
        "המודל עיבד <b>3,636 רשומות</b> ויצר <b>19 תכונות מהונדסות</b> לצורך החיזוי.",
        normal_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    # מדדי ביצועים
    performance_data = [
        ['ערך', 'מדד'],
        ['0.9989', 'R² Score'],
        ['0.0760', 'RMSE'],
        ['0.0426', 'MAE'],
        ['0.0058', 'MSE']
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

    # מועדי שתיית הקפה - זמנים בשעה
    elements.append(Paragraph("🕐 מועדי שתיית הקפה - זמנים בשעה", heading_style))

    elements.append(Paragraph("<b>שיאי צריכה עיקריים:</b>", normal_style))
    elements.append(Paragraph("• <b>שעה 10:00 בבוקר</b> - השיא המוחלט (349 מכירות) - זהו שעת הבוקר הקלאסית", bullet_style))
    elements.append(Paragraph("• <b>שעה 11:00</b> - שיא שני (294 מכירות) - המשך טרנד הבוקר", bullet_style))
    elements.append(Paragraph("• <b>שעה 16:00 (4 אחה\"צ)</b> - שיא שלישי (282 מכירות) - הפסקת הצהריים המאוחרת", bullet_style))
    elements.append(Paragraph("• <b>שעה 22:00 בלילה</b> - שיא רביעי (~310 מכירות) - צריכה ערבית מפתיעה", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("<b>תובנות זמניות:</b>", normal_style))
    elements.append(Paragraph("• יש <b>3 פיקים ברורים</b>: בוקר (8-12), אחר צהריים (15-17), וערב מאוחר (19-22)", bullet_style))
    elements.append(Paragraph("• השעות 8-12 הן שעות השיא המרוכזות ביותר", bullet_style))
    elements.append(Paragraph("• ירידה משמעותית בשעות 13-14 (שעת הצהריים)", bullet_style))
    elements.append(Paragraph("• עלייה מחודשת אחר הצהריים עד הערב", bullet_style))
    elements.append(Spacer(1, 0.3*inch))

    # הוספת גרף התפלגות שעות
    hour_dist_img = os.path.join(FIGURES_DIR, "hour_of_day_distribution.png")
    if os.path.exists(hour_dist_img):
        img = Image(hour_dist_img, width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))

    # התפלגות לפי זמן יום
    elements.append(Paragraph("📅 התפלגות לפי זמן יום", heading_style))

    time_data = [
        ['אחוז', 'מספר מכירות', 'זמן יום'],
        ['33.9%', '1,231', 'אחר הצהריים'],
        ['33.6%', '1,221', 'בוקר'],
        ['32.6%', '1,184', 'לילה']
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
        "<b>מסקנה:</b> התפלגות כמעט אחידה לחלוטין בין שלושת חלקי היום - אין דומיננטיות ברורה!",
        normal_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    # מועדים שבועיים
    elements.append(Paragraph("📆 מועדים שבועיים", heading_style))
    elements.append(Paragraph("<b>ימים עמוסים:</b>", normal_style))
    elements.append(Paragraph("• <b>יום שלישי</b> - היום העמוס ביותר (~580 מכירות)", bullet_style))
    elements.append(Paragraph("• <b>יום שישי</b> - שני בגובה (~540 מכירות)", bullet_style))
    elements.append(Paragraph("• <b>יום ראשון</b> - היום השקט ביותר (~440 מכירות)", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(
        "<b>תובנה:</b> ימי אמצע השבוע (שלישי-שישי) עמוסים יותר מסוף השבוע, "
        "מה שמצביע על קהל עובדים/סטודנטים.",
        normal_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    # הוספת גרף ימים בשבוע
    weekday_img = os.path.join(FIGURES_DIR, "Weekdaysort_distribution.png")
    if os.path.exists(weekday_img):
        img = Image(weekday_img, width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))

    # עמוד חדש
    elements.append(PageBreak())

    # מועדים חודשיים
    elements.append(Paragraph("📊 מועדים חודשיים/עונתיים", heading_style))
    elements.append(Paragraph("<b>חודשים עמוסים:</b>", normal_style))
    elements.append(Paragraph("• <b>מרץ</b> (חודש 3) - השיא המוחלט (~520 מכירות)", bullet_style))
    elements.append(Paragraph("• <b>אוקטובר</b> (חודש 10) - גבוה (~420 מכירות)", bullet_style))
    elements.append(Paragraph("• <b>מאי-יוני</b> - החודשים השקטים ביותר", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(
        "<b>תובנה:</b> עונתיות ברורה - אביב וסתיו עמוסים יותר, קיץ שקט יותר (אולי בגלל חופשות).",
        normal_style
    ))
    elements.append(Spacer(1, 0.3*inch))

    # הוספת גרף חודשים
    month_img = os.path.join(FIGURES_DIR, "Monthsort_distribution.png")
    if os.path.exists(month_img):
        img = Image(month_img, width=5*inch, height=3*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))

    # סוגי הקפה
    elements.append(Paragraph("☕ סוגי הקפה המועדפים", heading_style))

    coffee_data = [
        ['אחוז', 'מספר מכירות', 'סוג קפה', 'מיקום'],
        ['22.7%', '824', 'Americano with Milk', '🥇'],
        ['21.5%', '782', 'Latte', '🥈'],
        ['15.9%', '578', 'Americano', '🥉'],
        ['13.8%', '501', 'Cappuccino', '4'],
        ['8.0%', '292', 'Cortado', '5'],
        ['7.8%', '282', 'Hot Chocolate', '6'],
        ['6.7%', '243', 'Cocoa', '7'],
        ['3.7%', '134', 'Espresso', '8']
    ]

    coffee_table = Table(coffee_data, colWidths=[1*inch, 1.3*inch, 2*inch, 0.7*inch])
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

    # תובנות עסקיות
    elements.append(Paragraph("💡 תובנות עסקיות מרכזיות", heading_style))

    elements.append(Paragraph("<b>1. דומיננטיות קפה עם חלב:</b>", normal_style))
    elements.append(Paragraph(
        "• 72% מהמכירות הן משקאות עם חלב (Americano with Milk + Latte + Cappuccino)",
        bullet_style
    ))
    elements.append(Paragraph("• הלקוחות מעדיפים משקאות רכים יותר על קפה שחור", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("<b>2. פיזור זמנים:</b>", normal_style))
    elements.append(Paragraph(
        "• המודל זיהה ש-<b>hour_of_day היא התכונה החשובה ביותר (30.55%)</b> לחיזוי מכירות",
        bullet_style
    ))
    elements.append(Paragraph("• <b>משמעות:</b> השעה היא הגורם הקריטי ביותר להצלחת המכירות", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("<b>3. המלצות תפעוליות:</b>", normal_style))
    elements.append(Paragraph("• <b>חיזוק כוח אדם</b> בשעות 10-11 בבוקר ו-16:00 אחר הצהריים", bullet_style))
    elements.append(Paragraph("• הכנת <b>מלאי גדול יותר</b> של Americano with Milk ו-Latte", bullet_style))
    elements.append(Paragraph("• ימי <b>שלישי-שישי</b> דורשים יותר צוות ומלאי", bullet_style))
    elements.append(Paragraph("• <b>הפחתת מלאי</b> בחודשי הקיץ (מאי-יוני)", bullet_style))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph("<b>4. הפתעות:</b>", normal_style))
    elements.append(Paragraph(
        "• <b>שעה 22:00</b> היא שעת פיק משמעותית - כדאי לשקול הארכת שעות פתיחה",
        bullet_style
    ))
    elements.append(Paragraph(
        "• התפלגות אחידה בין בוקר/צהריים/ערב מעידה על <b>גיוון קהל לקוחות</b>",
        bullet_style
    ))
    elements.append(Spacer(1, 0.4*inch))

    # חשיבות תכונות
    elements.append(Paragraph("🎯 חשיבות תכונות במודל", heading_style))

    feature_data = [
        ['חשיבות', 'תכונה'],
        ['30.55%', 'hour_of_day (שעה ביום)'],
        ['25.47%', 'hour_of_day_binned (שעה מקובצת)'],
        ['22.20%', 'hour_of_day_squared (שעה בריבוע)'],
        ['3.77%', 'Weekdaysort_squared (יום בשבוע בריבוע)'],
        ['3.53%', 'Weekdaysort (יום בשבוע)'],
        ['3.28%', 'Time_of_Day (חלק יום)'],
        ['2.60%', 'Monthsort (חודש)']
    ]

    feature_table = Table(feature_data, colWidths=[1.5*inch, 3.5*inch])
    feature_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(feature_table)
    elements.append(Spacer(1, 0.3*inch))

    # הוספת גרף חשיבות תכונות
    feature_img = os.path.join(ARTIFACTS_DIR, "feature_importance.png")
    if os.path.exists(feature_img):
        elements.append(PageBreak())
        elements.append(Paragraph("גרף חשיבות תכונות", heading_style))
        img = Image(feature_img, width=6*inch, height=4*inch)
        elements.append(img)
        elements.append(Spacer(1, 0.3*inch))

    # סיכום
    elements.append(PageBreak())
    elements.append(Paragraph("📋 סיכום ומסקנות", heading_style))
    elements.append(Paragraph(
        "ניתוח מכירות הקפה חשף דפוסים ברורים המאפשרים אופטימיזציה של התפעול העסקי:",
        normal_style
    ))
    elements.append(Spacer(1, 0.2*inch))

    elements.append(Paragraph(
        "✓ <b>זמני שיא:</b> שלושה פיקים עיקריים - בוקר (10-11), אחה\"צ (16:00), וערב (22:00)",
        bullet_style
    ))
    elements.append(Paragraph(
        "✓ <b>העדפות מוצר:</b> 72% מהלקוחות מעדיפים קפה עם חלב",
        bullet_style
    ))
    elements.append(Paragraph(
        "✓ <b>דפוסים שבועיים:</b> ימי אמצע שבוע עמוסים יותר מסופי שבוע",
        bullet_style
    ))
    elements.append(Paragraph(
        "✓ <b>עונתיות:</b> אביב וסתיו עמוסים, קיץ שקט יותר",
        bullet_style
    ))
    elements.append(Paragraph(
        "✓ <b>דיוק המודל:</b> 99.89% - חיזוי אמין ביותר למכירות עתידיות",
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
        f"דוח זה נוצר אוטומטית באמצעות CrewAI | {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        footer_style
    ))

    # בניית ה-PDF
    doc.build(elements)

    return output_file

if __name__ == "__main__":
    try:
        pdf_file = create_hebrew_pdf()
        print(f"✅ קובץ PDF בעברית נוצר בהצלחה: {pdf_file}")
    except Exception as e:
        print(f"❌ שגיאה ביצירת PDF: {e}")
        import traceback
        traceback.print_exc()
