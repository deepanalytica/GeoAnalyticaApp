import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import requests

# Download the checkpoint if it doesn't exist
def download_checkpoint(checkpoint_url, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        with st.spinner("Descargando el modelo de segmentación..."):
            response = requests.get(checkpoint_url)
            with open(checkpoint_path, 'wb') as f:
                f.write(response.content)

checkpoint_url = "https://dl.fbaipublicfiles.com/segment-anything/sam_vit_h_4b8939.pth"
checkpoint_path = "sam_vit_h_4b8939.pth"
download_checkpoint(checkpoint_url, checkpoint_path)

# Load the SAM model and mask generator
@st.cache_resource
def load_model():
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

# Segment the image using SAM
def segment_image(mask_generator, image):
    masks = mask_generator.generate(image)
    return masks

# Draw the segmentation masks on the image
def draw_masks(image, masks):
    output_image = image.copy()
    for mask in masks:
        color = np.random.randint(0, 255, (3,)).tolist()
        for segment in mask['segments']:
            cv2.polylines(output_image, [np.array(segment).astype(np.int32)], isClosed=True, color=color, thickness=2)
    return output_image

# Streamlit app
def main():
    st.title("Clasificación de Sondajes Mineros con Segment Anything")
    uploaded_file = st.file_uploader("Sube una imagen de sondaje", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Segment the image
        mask_generator = load_model()
        with st.spinner("Segmentando la imagen..."):
            masks = segment_image(mask_generator, image)

        # Draw masks and display results
        segmented_image = draw_masks(image, masks)
        st.image(image, caption='Imagen Original', use_column_width=True)
        st.image(segmented_image, caption='Imagen Segmentada', use_column_width=True)

if __name__ == "__main__":
    main()
