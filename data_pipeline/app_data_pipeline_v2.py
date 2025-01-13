import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Initialize session state variables
if 'show_first_five' not in st.session_state:
    st.session_state.show_first_five = False

if 'species_selected' not in st.session_state:
    st.session_state.species_selected = False

if 'export_clicked' not in st.session_state:
    st.session_state.export_clicked = False

# Load Iris dataset
@st.cache
def load_data():
    return sns.load_dataset('iris')

# Data visualization
def plot_data(data):
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='sepal_length', y='sepal_width', hue='species', style='species', ax=ax)
    plt.legend(title='Species')
    return fig

# Export data
def export_data(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download CSV File</a>'
    return href

# Streamlit UI
def app():
    st.title('Iris Data Analysis Pipeline')
    data = load_data()

    # Button to display the first five rows
    if st.button('Display First Five Rows'):
        st.session_state.show_first_five = True

    if st.session_state.show_first_five:
        st.write("First Five Rows of the Dataset:")
        st.write(data.head())

    # Selectbox to filter species
    species_option = st.selectbox('Filter species', ['All'] + list(data['species'].unique()))
    if species_option:
        st.session_state.species_selected = True
        filtered_data = data[data['species'] == species_option] if species_option != 'All' else data
    
    if st.session_state.species_selected:
        # Visualization section
        st.write("Sepal Length vs. Sepal Width by Species")
        fig = plot_data(filtered_data)
        st.pyplot(fig)

        # Analysis section
        st.write("Statistical Analysis")
        st.write(filtered_data.describe())

    # Export section
    if st.button('Export Filtered Data'):
        st.session_state.export_clicked = True
    
    if st.session_state.export_clicked:
        href = export_data(filtered_data)
        st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    app()
