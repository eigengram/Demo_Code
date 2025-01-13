import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Load Iris dataset
@st.cache
def load_data():
    return sns.load_dataset('iris')

# Data cleaning (In this simple example, it's just filtering by species)
def clean_data(data, species):
    if species != 'All':
        return data[data['species'] == species]
    return data

# Data visualization
def plot_data(data):
    st.write("Sepal Length vs. Sepal Width by Species")
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='sepal_length', y='sepal_width', hue='species', style='species', ax=ax)
    plt.legend(title='Species')
    st.pyplot(fig)

# Basic Data Analysis (mean of features)
def analyze_data(data):
    return data.describe()

# Export data
def export_data(data):
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

# Streamlit UI
def app():
    st.title('Iris Data Analysis Pipeline')
    data = load_data()

    if st.button('Display First Five Rows'):
        st.write(data.head())

    species_option = st.selectbox('Filter species', ['All'] + list(data['species'].unique()))
    filtered_data = clean_data(data, species_option)

    if st.button('Visualize Data'):
        plot_data(filtered_data)

    if st.button('Show Analysis'):
        st.write(analyze_data(filtered_data))

    if st.button('Export Data'):
        export_data(filtered_data)

if __name__ == "__main__":
    app()
