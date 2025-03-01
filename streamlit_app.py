import streamlit as st
import numpy as np
import cv2
from PIL import Image
import scipy.signal
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings

def apply_filter(image, kernel):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    st.write("### Step 1: Convert to Grayscale")
    st.image(image, caption="Grayscale Image", use_column_width=True)
    
    st.write("### Step 2: Kernel Used for Convolution")
    st.write(kernel)
    
    st.write("### Step 3: Applying Convolution")
    filtered_image = scipy.signal.convolve2d(image, kernel, mode='same', boundary='symm')
    
    st.write("### Step 4: Normalizing the Output")
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    st.write("### Step 5: Intermediate Visualization")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original Grayscale Image")
    ax[1].imshow(kernel, cmap='gray')
    ax[1].set_title("Kernel")
    ax[2].imshow(filtered_image, cmap='gray')
    ax[2].set_title("Filtered Image")
    
    st.pyplot(fig)
    return filtered_image

st.title("Kernel Filter Visualizer with Intermediate Steps")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

st.write("### Enter Kernel Values")
kernel_size = st.number_input("Kernel Size (NxN, must be odd)", min_value=1, max_value=9, value=3, step=2)
kernel = np.zeros((kernel_size, kernel_size))

cols = st.columns(kernel_size)
for i in range(kernel_size):
    for j in range(kernel_size):
        kernel[i, j] = cols[j].number_input(f"({i},{j})", value=0.0)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    st.write("### Original Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Apply Filter"):
        filtered_image = apply_filter(image, kernel)
        st.write("### Final Filtered Image")
        st.image(filtered_image, caption="Filtered Image", use_column_width=True)
