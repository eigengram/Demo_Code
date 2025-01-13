import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import StringIO
from contextlib import redirect_stdout

# Ensure 'data' and 'code_blocks' are initialized in the session state
if 'data' not in st.session_state:
    st.session_state.data = sns.load_dataset('iris')  # Load Iris dataset initially

if 'code_blocks' not in st.session_state:
    st.session_state.code_blocks = []

# Function to execute user's custom Python code and capture both its output and any changes to 'data'
def execute_and_capture(code, data):
    local_ns = {"pd": pd, "sns": sns, "plt": plt, "data": data}
    try:
        # Capture standard output
        with StringIO() as buf, redirect_stdout(buf):
            exec(code, globals(), local_ns)
            return local_ns["data"], buf.getvalue()  # Return the potentially modified 'data'
    except Exception as e:
        return data, str(e)  # Return the input 'data' unmodified in case of an error

# Streamlit UI
def app():
    st.title('Iris Data Analysis Pipeline')

    st.write("First Five Rows of the Dataset:")
    st.write(st.session_state.data.head())

    # Section to add and execute custom code blocks iteratively
    st.markdown("### Add and Execute Custom Code")
    st.markdown("⚠️ **Caution:** Executing custom code can be risky. Use with caution.")
    code = st.text_area("Enter your Python code here. Use 'data' as the DataFrame variable.", 
                        value="", height=100, key="code_input")

    if st.button('Execute Code'):
        # Execute the code and capture any output or modifications to 'data'
        st.session_state.data, output = execute_and_capture(code, st.session_state.data)
        st.session_state.code_blocks.append((code, output))
        st.experimental_rerun()

    # Display previously executed code blocks and their outputs
    for i, (code, output) in enumerate(st.session_state.code_blocks, start=1):
        st.text_area(f"Code Block {i}", value=code, height=100, key=f"code_{i}")
        st.text(output)

    # Option to clear all executed code blocks
    if st.button('Clear All Code Blocks'):
        st.session_state.code_blocks = []
        st.experimental_rerun()

    # Visualization and Export Data sections can also use 'st.session_state.data'
    # For example, to visualize the current state of 'data':
    if st.button('Visualize Data'):
        fig, ax = plt.subplots()
        sns.scatterplot(data=st.session_state.data, x='sepal_length', y='sepal_width', hue='species', style='species', ax=ax)
        plt.legend(title='Species')
        st.pyplot(fig)

    # Export data function remains unchanged
    # Add export button and logic as needed, similar to previous examples

if __name__ == "__main__":
    app()
