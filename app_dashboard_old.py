import streamlit as st
import subprocess
import os
import time

def main():
    st.title("App Dashboard")
    st.write("Welcome to the app dashboard. Select an app from the dropdown list.")

    # Get the absolute path of the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Dictionary mapping app names to their script paths, directories, and ports
    apps = {
        "Text Search Vector DB": {"path": os.path.join(current_dir, "text_search_vector_db", "app.py"), "dir": os.path.join(current_dir, "text_search_vector_db"), "port": 8502},
        "Breast Cancer": {"path": os.path.join(current_dir, "breast_cancer_ultrasound", "breast_app.py"), "dir": os.path.join(current_dir, "breast_cancer_ultrasound"), "port": 8503},
        "Clip Text to Image Search": {"path": os.path.join(current_dir, "clip_text_to_image", "app_clip_text_to_image.py"), "dir": os.path.join(current_dir, "clip_text_to_image"), "port": 8504},
        "Data Pipeline with custom code (multi)": {"path": os.path.join(current_dir, "data_pipeline", "app_data_pipeline_with_user_custom_code.py"), "dir": os.path.join(current_dir, "data_pipeline"), "port": 8505},
        "Drug Discovery Classification": {"path": os.path.join(current_dir, "drug_discovery_classification_cyp450", "app_cyp450.py"), "dir": os.path.join(current_dir, "drug_discovery_classification_cyp450"), "port": 8506},
        "Lab Parameters Entry": {"path": os.path.join(current_dir, "lab_parameters_entry", "app_add_lab_parameter_detail.py"), "dir": os.path.join(current_dir, "lab_parameters_entry"), "port": 8507},
        "Merge PDF": {"path": os.path.join(current_dir, "merge_pdf", "app_merge_pdf.py"), "dir": os.path.join(current_dir, "merge_pdf"), "port": 8508},
        "No Code Platform": {"path": os.path.join(current_dir, "no_code_platform", "app_no_code.py"), "dir": os.path.join(current_dir, "no_code_platform"), "port": 8509},
        "Reverse Image Search Patents": {"path": os.path.join(current_dir, "reverse_image_search_patents", "app_reverse_image_search.py"), "dir": os.path.join(current_dir, "reverse_image_search_patents"), "port": 8510},
        "Structured Data Heart Prediction": {"path": os.path.join(current_dir, "structured_data_heart_prediction", "app_structured_heart_predict.py"), "dir": os.path.join(current_dir, "structured_data_heart_prediction"), "port": 8511}
    }

    # Select box for navigation
    options = ["Select"] + list(apps.keys())
    selected_app = st.selectbox("Choose an app", options)

    # Load and execute the selected app
    if selected_app and selected_app != "Select":
        app_info = apps[selected_app]
        script_path = app_info["path"]
        working_dir = app_info["dir"]
        port = app_info["port"]
        try:
            # Use subprocess to run the selected app script with the correct working directory and port
            st.write(f"Running command: streamlit run {script_path} --server.port {port}")
            process = subprocess.Popen(['streamlit', 'run', script_path, '--server.port', str(port)], cwd=working_dir, shell=True)
            st.write(f"{selected_app} is running at [http://localhost:{port}](http://localhost:{port})")
            
            # Optional: Wait for the server to start
            time.sleep(5)  # Wait for 5 seconds before trying to access the link
            
            st.success(f"{selected_app} started successfully.")
        except Exception as e:
            st.error(f"An error occurred while running the app: {e}")

# Run the app
if __name__ == "__main__":
    main()
