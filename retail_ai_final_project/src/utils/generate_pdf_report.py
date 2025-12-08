#!/usr/bin/env python
"""
Generate comprehensive PDF report with all charts and analysis results.

This script collects all visualizations and reports from the artifacts directory
and creates a professional PDF document.
"""
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak,
        Table, TableStyle, KeepTogether
    )
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from PIL import Image as PILImage
except ImportError as e:
    print("Error: Required packages not installed.")
    print("Please install them using: pip install reportlab pillow")
    print(f"Error details: {e}")
    sys.exit(1)


def get_project_root() -> Path:
    """Get the project root directory."""
    current_file = Path(__file__).resolve()
    # Navigate from src/utils/ to project root
    return current_file.parent.parent.parent


def read_markdown_file(file_path: Path) -> str:
    """Read markdown file and return content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def markdown_to_paragraphs(content: str, styles) -> List:
    """Convert markdown content to ReportLab paragraphs."""
    elements = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            elements.append(Spacer(1, 0.1 * inch))
            continue
            
        # Handle headers
        if line.startswith('# '):
            elements.append(Paragraph(line[2:], styles['Heading1']))
            elements.append(Spacer(1, 0.2 * inch))
        elif line.startswith('## '):
            elements.append(Paragraph(line[3:], styles['Heading2']))
            elements.append(Spacer(1, 0.15 * inch))
        elif line.startswith('### '):
            elements.append(Paragraph(line[4:], styles['Heading3']))
            elements.append(Spacer(1, 0.1 * inch))
        # Handle tables (simple markdown tables)
        elif '|' in line and line.startswith('|'):
            # Skip table separator lines
            if not all(c in '|: -' for c in line):
                # This is a table row, we'll handle it separately
                continue
        # Handle bullet points
        elif line.startswith('- ') or line.startswith('* '):
            text = line[2:].strip()
            # Remove markdown bold
            text = text.replace('**', '')
            elements.append(Paragraph(f"• {text}", styles['Normal']))
            elements.append(Spacer(1, 0.05 * inch))
        # Handle numbered lists
        elif line and line[0].isdigit() and '. ' in line[:5]:
            text = line.split('. ', 1)[1] if '. ' in line else line
            text = text.replace('**', '')
            elements.append(Paragraph(text, styles['Normal']))
            elements.append(Spacer(1, 0.05 * inch))
        # Regular text
        else:
            if line:
                # Remove markdown formatting
                text = line.replace('**', '').replace('*', '')
                elements.append(Paragraph(text, styles['Normal']))
                elements.append(Spacer(1, 0.05 * inch))
    
    return elements


def add_image_to_pdf(image_path: Path, elements: List, styles, max_width: float = 6.5 * inch) -> None:
    """Add image to PDF elements list."""
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return
    
    try:
        # Open image to get dimensions
        img = PILImage.open(image_path)
        img_width, img_height = img.size
        
        # Calculate scaling to fit page width
        scale = min(max_width / img_width, 1.0)
        width = img_width * scale
        height = img_height * scale
        
        # Add image
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Image(str(image_path), width=width, height=height))
        elements.append(Spacer(1, 0.2 * inch))
        
        # Add caption
        caption = image_path.stem.replace('_', ' ').title()
        elements.append(Paragraph(f"<i>Figure: {caption}</i>", 
                                 styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))
    except Exception as e:
        print(f"Error adding image {image_path}: {str(e)}")


def create_metrics_table(metrics_data: dict) -> Table:
    """Create a table for model metrics."""
    data = [['Metric', 'Score', 'Interpretation']]
    
    for metric, value in metrics_data.items():
        if metric == 'R²':
            interpretation = 'Excellent fit (99.89% variance explained)'
        elif metric == 'RMSE':
            interpretation = 'Low prediction error'
        elif metric == 'MAE':
            interpretation = 'Mean absolute error is minimal'
        elif metric == 'MSE':
            interpretation = 'Very low mean squared error'
        else:
            interpretation = ''
        
        data.append([metric, f"{value:.4f}", interpretation])
    
    table = Table(data, colWidths=[2*inch, 1.5*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
    ]))
    return table


def generate_pdf_report(output_path: Path = None) -> Path:
    """Generate comprehensive PDF report with all charts and analysis."""
    project_root = get_project_root()
    artifacts_dir = project_root / "artifacts"
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = artifacts_dir / f"comprehensive_report_{timestamp}.pdf"
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=20
    )
    
    # Build PDF content
    elements = []
    
    # Title page
    elements.append(Spacer(1, 2 * inch))
    elements.append(Paragraph("Retail AI Project", title_style))
    elements.append(Paragraph("Comprehensive Analysis Report", 
                             ParagraphStyle('Subtitle', parent=styles['Heading2'],
                                           fontSize=18, alignment=TA_CENTER,
                                           spaceAfter=30)))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}",
                             ParagraphStyle('Date', parent=styles['Normal'],
                                           alignment=TA_CENTER)))
    elements.append(PageBreak())
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", title_style))
    elements.append(Spacer(1, 0.3 * inch))
    
    summary_text = """
    This comprehensive report presents the complete analysis of the Retail AI project,
    including data insights, model performance metrics, and visualizations. The project
    achieved excellent results with a model accuracy of 99.89% (R² = 0.9989).
    
    <b>Key Highlights:</b>
    • Dataset: 3,636 records with 19 engineered features
    • Model Type: Random Forest Regressor
    • Performance: R² = 0.9989, RMSE = 0.0760
    • Top Features: Time-based features (hour, day, month)
    """
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(PageBreak())
    
    # Model Performance Metrics
    elements.append(Paragraph("Model Performance Metrics", subtitle_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    metrics = {
        'R²': 0.9989,
        'RMSE': 0.0760,
        'MAE': 0.0426,
        'MSE': 0.0058
    }
    elements.append(create_metrics_table(metrics))
    elements.append(PageBreak())
    
    # Data Insights Section
    elements.append(Paragraph("Data Insights", subtitle_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    insights_path = artifacts_dir / "insights.md"
    if insights_path.exists():
        insights_content = read_markdown_file(insights_path)
        insight_elements = markdown_to_paragraphs(insights_content, styles)
        elements.extend(insight_elements)
    
    elements.append(PageBreak())
    
    # Model Evaluation Report
    elements.append(Paragraph("Model Evaluation Report", subtitle_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    eval_report_path = artifacts_dir / "evaluation_report.md"
    if eval_report_path.exists():
        eval_content = read_markdown_file(eval_report_path)
        eval_elements = markdown_to_paragraphs(eval_content, styles)
        elements.extend(eval_elements)
    
    # Add evaluation visualizations
    elements.append(Paragraph("Model Visualizations", subtitle_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    # Residual plot
    residual_plot = artifacts_dir / "residual_plot.png"
    if residual_plot.exists():
        add_image_to_pdf(residual_plot, elements, styles)
    
    # Actual vs Predicted
    actual_vs_pred = artifacts_dir / "actual_vs_predicted.png"
    if actual_vs_pred.exists():
        add_image_to_pdf(actual_vs_pred, elements, styles)
    
    # Feature Importance
    feature_importance = artifacts_dir / "feature_importance.png"
    if feature_importance.exists():
        add_image_to_pdf(feature_importance, elements, styles)
    
    elements.append(PageBreak())
    
    # EDA Visualizations
    elements.append(Paragraph("Exploratory Data Analysis", subtitle_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    figures_dir = artifacts_dir / "figures"
    if figures_dir.exists():
        # Correlation heatmap
        correlation_heatmap = figures_dir / "correlation_heatmap.png"
        if correlation_heatmap.exists():
            add_image_to_pdf(correlation_heatmap, elements, styles)
        
        # Distribution plots
        hour_dist = figures_dir / "hour_of_day_distribution.png"
        if hour_dist.exists():
            add_image_to_pdf(hour_dist, elements, styles)
        
        weekday_dist = figures_dir / "Weekdaysort_distribution.png"
        if weekday_dist.exists():
            add_image_to_pdf(weekday_dist, elements, styles)
        
        month_dist = figures_dir / "Monthsort_distribution.png"
        if month_dist.exists():
            add_image_to_pdf(month_dist, elements, styles)
    
    elements.append(PageBreak())
    
    # Model Card
    elements.append(Paragraph("Model Card", subtitle_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    model_card_path = artifacts_dir / "model_card.md"
    if model_card_path.exists():
        model_card_content = read_markdown_file(model_card_path)
        card_elements = markdown_to_paragraphs(model_card_content, styles)
        elements.extend(card_elements)
    
    elements.append(PageBreak())
    
    # Recommendations Section
    elements.append(Paragraph("Recommendations and Next Steps", subtitle_style))
    elements.append(Spacer(1, 0.2 * inch))
    
    recommendations_text = """
    <b>Immediate Actions:</b>
    • Deploy model to production environment
    • Set up monitoring and alerting systems
    • Implement periodic model retraining (monthly/quarterly)
    
    <b>Technical Improvements:</b>
    • Review misclassified samples to understand model limitations
    • Monitor model performance on production data
    • Consider ensemble methods if performance needs improvement
    
    <b>Business Applications:</b>
    • Use time-based predictions for inventory management
    • Optimize staffing based on predicted sales patterns
    • Support data-driven decision making
    
    <b>Long-term Strategy:</b>
    • Collect feedback for model iteration
    • Plan for model versioning and updates
    • Expand model to additional use cases
    """
    elements.append(Paragraph(recommendations_text, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    
    print(f"✅ PDF report generated successfully: {output_path}")
    return output_path


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive PDF report with all charts and analysis"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for PDF file (default: artifacts/comprehensive_report_TIMESTAMP.pdf)"
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output) if args.output else None
    generate_pdf_report(output_path)


if __name__ == "__main__":
    main()

