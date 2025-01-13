from transformers import CLIPProcessor, CLIPModel
import os

def download_clip_files(model_name="openai/clip-vit-base-patch32", save_directory="./clip_model"):
    # Ensure the save directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Download and save the model
    model = CLIPModel.from_pretrained(model_name)
    model.save_pretrained(save_directory)
    
    # Download and save the processor
    processor = CLIPProcessor.from_pretrained(model_name)
    processor.save_pretrained(save_directory)

    print(f"CLIP model and processor files have been saved to {save_directory}")

# Specify the model name and the directory where you want to save the files
model_name = "openai/clip-vit-base-patch32"
save_directory = "./clip_model"

# Run the function to download the files
download_clip_files(model_name, save_directory)
