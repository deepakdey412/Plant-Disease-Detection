# Library imports
import numpy as np
import streamlit as st
import cv2
import tensorflow as tf

# Load models for Apple, Potato, Tulsi, Tomato, and Rose with error handling
try:
    apple_model = tf.keras.models.load_model("apple_disease_model.h5", compile=False)
    st.write("Apple model loaded successfully.")
except Exception as e:
    apple_model = None
    st.error(f"Failed to load Apple model: {e}")

try:
    potato_model = tf.keras.models.load_model("potato_disease_model.h5", compile=False)
    st.write("Potato model loaded successfully.")
except Exception as e:
    potato_model = None
    st.error(f"Failed to load Potato model: {e}")

try:
    tulsi_model = tf.keras.models.load_model("tulsi_disease_model.h5", compile=False)
    st.write("Tulsi model loaded successfully.")
except Exception as e:
    tulsi_model = None
    st.error(f"Failed to load Tulsi model: {e}")

try:
    tomato_model = tf.keras.models.load_model("tomato_disease_model.h5", compile=False)
    st.write("Tomato model loaded successfully.")
except Exception as e:
    tomato_model = None
    st.error(f"Failed to load Tomato model: {e}")

try:
    rose_model = tf.keras.models.load_model("rose_disease_model.h5", compile=False)
    st.write("Rose model loaded successfully.")
except Exception as e:
    rose_model = None
    st.error(f"Failed to load Rose model: {e}")

# Class Names for Each Model
APPLE_CLASS_NAMES = ['Apple_black_rot', 'Apple_cedar_rust', 'Apple_scab']
POTATO_CLASS_NAMES = ['Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight']
TULSI_CLASS_NAMES = ['Tulsi_fungal', 'Tulsi_healthy']
TOMATO_CLASS_NAMES = ['Tomato___healthy', 'Tomato___Septoria_leaf_spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
ROSE_CLASS_NAMES = ['Healthy_Leaf_Rose', 'Rose_sawfly_Rose_slug']

# Streamlit App Title and Instructions
st.title("Plant Disease Detection")
st.header("Detect diseases in Apple, Potato, Tulsi, Tomato, and Rose plants")
st.markdown("Upload an image of the plant leaf to identify its condition.")

# Dropdown to select the plant type
plant_type = st.selectbox("Select the type of plant:", ("Apple", "Potato", "Tulsi", "Tomato", "Rose"))

# Uploading the plant image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
submit = st.button("Predict Disease")

# Prediction Logic
if submit:
    if plant_image is not None:
        try:
            # Read the uploaded image as an OpenCV image
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            
            # Displaying the uploaded image
            st.image(opencv_image, channels="BGR", caption="Uploaded Image")
            
            # Preprocessing: Resize image and normalize pixel values
            resized_image = cv2.resize(opencv_image, (256, 256))
            normalized_image = resized_image / 255.0  # Normalize to [0, 1]
            input_image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
            
            # Predict using the selected model
            if plant_type == "Apple" and apple_model is not None:
                st.subheader("Apple Disease Model Prediction:")
                predictions = apple_model.predict(input_image)
                result = APPLE_CLASS_NAMES[np.argmax(predictions)]
                st.success(f"Prediction: **{result.replace('_', ' ')}**")
            elif plant_type == "Potato" and potato_model is not None:
                st.subheader("Potato Disease Model Prediction:")
                predictions = potato_model.predict(input_image)
                result = POTATO_CLASS_NAMES[np.argmax(predictions)]
                st.success(f"Prediction: **{result.replace('_', ' ')}**")
            elif plant_type == "Tulsi" and tulsi_model is not None:
                st.subheader("Tulsi Disease Model Prediction:")
                predictions = tulsi_model.predict(input_image)
                result = TULSI_CLASS_NAMES[np.argmax(predictions)]
                st.success(f"Prediction: **{result.replace('_', ' ')}**")
            elif plant_type == "Tomato" and tomato_model is not None:
                st.subheader("Tomato Disease Model Prediction:")
                predictions = tomato_model.predict(input_image)
                result = TOMATO_CLASS_NAMES[np.argmax(predictions)]
                st.success(f"Prediction: **{result.replace('_', ' ')}**")
            elif plant_type == "Rose" and rose_model is not None:
                st.subheader("Rose Disease Model Prediction:")
                predictions = rose_model.predict(input_image)
                result = ROSE_CLASS_NAMES[np.argmax(predictions)]
                st.success(f"Prediction: **{result.replace('_', ' ')}**")
            else:
                st.error(f"Model for {plant_type} is not available.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Please upload an image before predicting.")
