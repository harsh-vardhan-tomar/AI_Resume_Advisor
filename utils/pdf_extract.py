"""
STEP 1: Extracting text from PDFs
-----------------------------------
This is the most basic step. Both the resume and the job description
are PDF files. We need to extract plain text from them so the AI
can read them.

We use pdfplumber because resumes often have columns/tables, and
pdfplumber handles those better than pypdf.
"""

import pdfplumber


def extract_text_from_pdf(file_path_or_buffer):
    """
    Input: path to a PDF file, or a file object uploaded via Streamlit
    Output: the full extracted text as a single string

    How it works:
    1. Open the PDF
    2. Extract text from each page
    3. Concatenate everything into one string
    """
    text = ""
    with pdfplumber.open(file_path_or_buffer) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # a page can sometimes be empty
                text += page_text + "\n"
    return clean_text(text)


def clean_text(text):
    """
    Basic cleanup - remove extra spaces and blank lines.
    Giving the AI clean text produces better results.
    """
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]  # drop empty lines
    return "\n".join(lines)


# Quick manual test if you run this file directly
if __name__ == "__main__":
    sample_path = "sample_resume.pdf"
    result = extract_text_from_pdf(sample_path)
    print(result[:500])  # show the first 500 characters