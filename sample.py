import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Set custom title and favicon for the browser tab
st.set_page_config(page_title="Chart Classification App", page_icon="📊")

# Load the pre-trained model
model = tf.keras.models.load_model(r'/Users/jagruth/Downloads/vgg16.h5')

def preprocess_image(image):
    """
    Preprocess the uploaded image: 
    1. Resize the image to match the model input.
    2. Normalize the pixel values.
    3. Add batch dimension.
    """
    image = image.resize((670, 480))  # Resize to the input size expected by the model
    image = np.array(image) / 255.0   # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension for model input
    return image

# App Title and Description
st.title('Chart Classification with Deep Learning')
st.write('Upload an image of a chart, and the model will predict its type.')

# File uploader for the image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# If an image is uploaded, process and classify it
if uploaded_image is not None:
    image = Image.open(uploaded_image)  # Open the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)  # Display the uploaded image

    # Preprocess the image and make predictions
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions)

    # Chart class labels
    class_labels = ['Area', 'Heatmap', 'Horizontal Bar', 'Horizontal Interval', 'Line', 'Manhattan',
                    'Map', 'Pie', 'Scatter', 'Scatter-Line', 'Surface', 'Venn', 'Vertical Bar',
                    'Vertical Box', 'Vertical Interval']

    # Display the predicted chart type
    predicted_label = class_labels[predicted_class]
    st.write(f'**Predicted Chart Type**: {predicted_label}')
