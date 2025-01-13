import streamlit as st
import base64
from io import BytesIO

# Functions to generate HTML code snippets for different form elements
def generate_header_html(header_text):
    return f"<h1>{header_text}</h1>"

def generate_input_html(placeholder_text):
    return f'<input type="text" placeholder="{placeholder_text}">'

def generate_button_html(button_text):
    return f'<button type="submit">{button_text}</button>'

# Function to download code snippet
def get_download_link(filename, text):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" target="_blank">Download file</a>'
    return href

# Streamlit app interface
def app():
    st.title("No-Code HTML Generator")

    # Session state to store accumulated code
    if 'accumulated_code' not in st.session_state:
        st.session_state.accumulated_code = ""

    option = st.sidebar.selectbox("Choose an HTML element to generate:",
                                  ('Header', 'Input Field', 'Button'))

    if option == 'Header':
        header_text = st.sidebar.text_input("Header Text", "Sample Header")
        html_code = generate_header_html(header_text)

    elif option == 'Input Field':
        placeholder_text = st.sidebar.text_input("Placeholder Text", "Enter something...")
        html_code = generate_input_html(placeholder_text)

    elif option == 'Button':
        button_text = st.sidebar.text_input("Button Text", "Submit")
        html_code = generate_button_html(button_text)

    editable_code = st.text_area("Generated HTML Code", html_code, height=150)
    
    if st.button('Add to Code Section'):
        st.session_state.accumulated_code += editable_code + "\n"
        st.success("Code snippet added!")

    st.text_area("Accumulated Code", st.session_state.accumulated_code, height=250)

    # Exporting options
    file_type = st.selectbox("Select file type to export:", ['HTML', 'TXT', 'PY'])
    file_name = st.text_input("Enter filename (without extension):", "download")
    full_file_name = f"{file_name}.{file_type.lower()}"

    if st.button('Export Code'):
        tmp_download_link = get_download_link(full_file_name, st.session_state.accumulated_code)
        st.markdown(tmp_download_link, unsafe_allow_html=True)

if __name__ == "__main__":
    app()
