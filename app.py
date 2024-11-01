import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Load the saved model
model = load_model('my_fakeddit_model.h5')

# Define your image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    if image is None:
        # Return a zero array with 3 channels for missing images
        return np.zeros(target_size + (3,))
    image = image.convert('RGB')  # Convert to RGB
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    return image_array

# Streamlit app
st.title('Fakeddit Multimedia Fake News Detection')

# User inputs
title = st.text_input('News Title', 'Type a news title here')
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    display_image = Image.open(uploaded_image)
    st.image(display_image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and make a prediction
    processed_image = preprocess_image(display_image)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

    # Assuming you have a way to vectorize the input title similar to how the training data was prepared
    # For simplicity, here we just use a placeholder
    vectorized_title = np.zeros((1, 1000))  # Placeholder for the actual text vectorization process

    # Make a prediction
    if st.button("Get Prediction"):
        prediction = model.predict([vectorized_title, processed_image])
        st.write(prediction[0]) 
        st.write(f'Prediction: {"Real" if prediction[0][0] > 0.5 else "Fake"}')