from fpdf import FPDF
import io

class PDF(FPDF):
    def header(self):
        # Use a Unicode font for the header
        self.set_font("DejaVu", "B", 16)
        self.cell(0, 10, "LLM Validation Report", ln=True, align="C")
        self.ln(10)
        
    def progress_bar(self, x, y, width, height, score, fill_color=(0, 128, 0), bg_color=(220, 220, 220)):
        """
        Draws a progress bar at position (x, y) with given width and height.
        The bar is filled proportionally to `score` (between 0 and 1).
        """
        # Draw background
        self.set_fill_color(*bg_color)
        self.rect(x, y, width, height, style="F")
        # Draw filled portion
        fill_width = width * score
        self.set_fill_color(*fill_color)
        self.rect(x, y, fill_width, height, style="F")
        # Draw border
        self.set_draw_color(0, 0, 0)
        self.rect(x, y, width, height, style="D")

def generate_pdf_report(query: str, llm_response: str, metrics: dict, average_score: float, overall_valid: bool, perplexity_info: dict, response_time: float) -> bytes:
    """
    Generates a PDF report containing:
      - Query, LLM response, and response time.
      - Validation metrics with progress bars.
      - Perplexity validation details.
    Returns the PDF as bytes.
    """
    pdf = PDF()
    
    # Add Unicode fonts (ensure these TTF files are in the same folder as this file)
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
    
    pdf.add_page()
    pdf.set_font("DejaVu", "", 12)
    
    # Print Query, Response, and Response Time
    pdf.multi_cell(0, 10, f"Query:\n{query}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"LLM Response:\n{llm_response}")
    pdf.ln(5)
    pdf.cell(0, 10, f"Response Time: {response_time:.2f} seconds", ln=True)
    pdf.ln(10)
    
    # Validation Metrics Section
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Validation Metrics", ln=True)
    pdf.ln(5)
    pdf.set_font("DejaVu", "", 12)
    
    # Define progress bar dimensions
    max_bar_width = 100  # in mm
    bar_height = 5
    x_start = 10
    current_y = pdf.get_y()
    
    for key, value in metrics.items():
        if key == "context_similarity" and (value is None or value == 0):
            continue  # Skip if no context similarity score
        text_line = f"{key.capitalize()} Score: {value}"
        pdf.cell(0, 10, text_line, ln=True)
        current_y = pdf.get_y()
        if isinstance(value, (int, float)):
            pdf.progress_bar(x_start, current_y, max_bar_width, bar_height, score=value)
            pdf.ln(10)
        else:
            pdf.ln(5)
    
    pdf.cell(0, 10, f"Average Score: {average_score}", ln=True)
    pdf.cell(0, 10, f"Overall Validation: {'Valid' if overall_valid else 'Not Valid'}", ln=True)
    pdf.ln(10)
    
    # Perplexity Validation Section
    pdf.set_font("DejaVu", "B", 14)
    pdf.cell(0, 10, "Perplexity Validation", ln=True)
    pdf.ln(5)
    pdf.set_font("DejaVu", "", 12)
    pdf.cell(0, 10, f"Log-Normalized Perplexity Ratio: {perplexity_info.get('log_normalized_perplexity_ratio', 'N/A')}", ln=True)
    pdf.cell(0, 10, f"Perplexity Validation: {'Valid' if perplexity_info.get('valid') else 'Not Valid'}", ln=True)
    pdf.multi_cell(0, 10, f"Reason: {perplexity_info.get('reason', '')}")
    pdf.ln(10)
    
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()
