import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import io

def merge_pdfs(pdfs):
    """Merge multiple PDFs into a single PDF"""
    pdf_writer = PdfWriter()

    for pdf in pdfs:
        pdf_reader = PdfReader(io.BytesIO(pdf.getvalue()))
        for page in range(len(pdf_reader.pages)):
            pdf_writer.add_page(pdf_reader.pages[page])

    merged_pdf = io.BytesIO()
    pdf_writer.write(merged_pdf)
    merged_pdf.seek(0)

    return merged_pdf

st.title('PDF Merger')

uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type='pdf')
if uploaded_files:
    files_to_display = {f.name: f for f in uploaded_files}
    ordered_files = st.multiselect("Arrange the files in the order you want them to be merged (top to bottom)(You can first delete all and iteratively select in the order you want your files to be in):", 
                                   options=list(files_to_display.keys()), default=list(files_to_display.keys()))

    if st.button('Merge PDFs'):
        files_to_merge = [files_to_display[name] for name in ordered_files]
        merged_pdf = merge_pdfs(files_to_merge)

        st.download_button(
            label="Download Merged PDF",
            data=merged_pdf,
            file_name="merged.pdf",
            mime="application/pdf"
        )
