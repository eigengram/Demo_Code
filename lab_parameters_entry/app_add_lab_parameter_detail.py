import streamlit as st
import pandas as pd
import os

# Define the path for the TSV file
TSV_FILE_PATH = 'lab_parameter_differential_diagnosis.tsv'

# The hidden reference variable
reference = 'Differential Diagnosis by Laboratory Medicine by Vincent Marks'

# Initialize the form
with st.form("lab_parameter_form"):
    st.write("Enter the details for the lab parameter")

    # Form fields
    lab_parameter_name = st.text_input("Lab Parameter Name")
    
    category_options = ["Choose a category", "abbreviation", "synonyms", "Reference Range", "present in", "general details", "production", 
                        "increased values", "decreased values", "Target Organ", "Function", "Test Purpose", 
                        "Hypothalamic hormone", "Occurence", "Isoenzymes", "Increased Isoenzymes", "Decreased Isoenzymes",
                        "Inflammatory mediators", "Inhibitors", "Complement proteins", "Hemocoagulation proteins",
                        "Miscellaneous", "Cellular immune regulation", "Metal binding proteins",
                        "Repair and resolution"]
    category = st.selectbox("Category", options=category_options, index=0)
    
    # Sub Category as text input for flexibility
    sub_category = st.text_input("Sub Category")
    
    details = st.text_area("Details", height=100)

    # Form submission button
    submitted = st.form_submit_button("Submit")

    if submitted:
        # Clean up the text inputs to remove unwanted line breaks
        lab_parameter_name = lab_parameter_name.replace("\n", " ")
        sub_category = sub_category.replace("\n", " ")
        details = details.replace("\n", " ")
        
        # Ensure selection other than the placeholder is recorded, else leave blank
        category = category if category != "Choose a category" else ""
        
        # Determine the next ID
        if os.path.exists(TSV_FILE_PATH):
            df_existing = pd.read_csv(TSV_FILE_PATH, sep='\t')
            next_id = df_existing['ID'].max() + 1
        else:
            next_id = 1

        # DataFrame for the new entry
        new_data = {
            'ID': [next_id],
            'Lab Parameter Name': [lab_parameter_name],
            'Reference': [reference],
            'Category': [category],
            'Sub Category': [sub_category],
            'Details': [details]
        }
        new_entry_df = pd.DataFrame(new_data)
        
        # Append or create new TSV file
        if os.path.exists(TSV_FILE_PATH):
            new_entry_df.to_csv(TSV_FILE_PATH, mode='a', sep='\t', index=False, header=False)
        else:
            new_entry_df.to_csv(TSV_FILE_PATH, sep='\t', index=False)
            
        st.success("Data saved successfully!")

# Display only the last 5 rows of the stored data
if os.path.exists(TSV_FILE_PATH):
    st.write("Last 5 entries in TSV file:")
    df_display = pd.read_csv(TSV_FILE_PATH, sep='\t')
    st.dataframe(df_display.tail(5))
