import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model('multi_class_model.h5')

# Define the image size for model input
IMG_SIZE = (224, 224)

# Add custom CSS for aesthetics
st.markdown(
    """
    <style>
    .title {
        margin-top: 0px;
        color: #FF5733; /* Coral */
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .text {
        color: #EFA18A; /* Slate Gray */
        font-size: 20px;
        font-weight: italic;
        text-align: center;
        margin-bottom: 20px;
    }
    .uploaded-image {
        width: 100%;
        max-width: 500px;
        margin-bottom: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .prediction {
        color: #FF5733; /* Coral */
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    .confidence {
        color: #FF5600; /* Coral */
        font-size: 18px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Display the title
st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='text'>This web application utilizes a pre-trained deep learning model to predict the presence of Alzheimer's disease based on uploaded brain images.</h1>", unsafe_allow_html=True)

st.sidebar.title("Upload Image")
st.sidebar.markdown("Please upload a brain scan image.")

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Convert to grayscale and resize
    image = image.convert('L')  # Convert to grayscale
    image = image.resize(IMG_SIZE)  # Resize to model input size
    img_array = np.array(image)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Function to make predictions
def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100  # Confidence in percentage
    return predicted_idx, confidence

# Class labels
class_labels = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Display the file uploader
uploaded_file = st.sidebar.file_uploader(label="", type=['jpg', 'jpeg', 'png'])

# Make predictions and display the result
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict the result
    predicted_idx, confidence = predict(image)
    predicted_label = class_labels[predicted_idx]
    
    # Display the result
    st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
else:
    st.sidebar.write("Please upload an image.")
