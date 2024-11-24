import streamlit as st
from PIL import Image
import pytesseract
import pyttsx3
import os
import cv2
import numpy as np
import io
import base64
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from threading import Thread
from gtts import gTTS

# Initialize the text-to-speech engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('volume', 1)

# Set up Google Generative AI API
GEMINI_API_KEY = "AIzaSyCfo0BgLWlwUDsA0bTDHflo_jQ067AdSqg"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
chat_model = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-1.5-flash")

# Streamlit page settings
st.set_page_config(page_title="Building AI Powered Solution for Assisting Visually Impaired Individuals", layout="wide")
st.markdown("""
    <style>
        .css-1d391kg {background-color: #f0f0f0;}
        .css-10trblm {background-color: #9c27b0; color: white;}
        .css-1khdih3 {font-size: 18px; font-family: 'Arial', sans-serif;}
        .css-1kfgqdz {background-color: #ffffff; border-radius: 10px; padding: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; font-size: 16px;}
        .stButton>button:hover {background-color: #45a049;}
    </style>
""", unsafe_allow_html=True)

# Title and description of the app
st.title("Building AI Powered Solution for Assisting Visually Impaired Individuals 🚀👁️")
st.markdown("""
    This app uses AI technologies to assist visually impaired people by:
    - 🌍 **Real-Time Scene Understanding**
    - 🔠 **Text-to-Speech Conversion**
    - 🛑 **Object & Obstacle Detection**

    Upload an image and let the AI help with detailed descriptions, text-to-speech conversion, and obstacle detection. Your eyes, powered by AI!
""", unsafe_allow_html=True)

# Sidebar for app features
st.sidebar.title("Features")
st.sidebar.markdown("""
- Real-Time Scene Understanding 🖼️
- Text-to-Speech Conversion 🎙️
- Object & Obstacle Detection 🛑
""")
st.sidebar.markdown("**Upload an image to get started!**")

# Function to extract text from the image using OCR
@st.cache_data
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Function to clean text (remove unnecessary spaces and newlines)
def clean_text(text):
    cleaned_text = ' '.join(text.splitlines())
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

# Function for converting text to speech
def text_to_speech(text):
    try:
        cleaned_text = clean_text(text)
        def speak():
            engine.say(cleaned_text)
            engine.runAndWait()
        tts_thread = Thread(target=speak)
        tts_thread.start()
        tts_thread.join()

    except Exception as e:
        st.error(f"Error with pyttsx3 TTS: {e}")
        tts = gTTS(cleaned_text)
        tts.save("output.mp3")
        os.system("start output.mp3")

# Function to preprocess the image for better OCR results
@st.cache_data
def preprocess_image(_image):
    image_np = np.array(_image)
    if len(image_np.shape) == 3:
        image_np = np.dot(image_np[...,:3], [0.2989, 0.5870, 0.1140])  # Convert to grayscale
    _, thresh_image = cv2.threshold(image_np.astype(np.uint8), 150, 255, cv2.THRESH_BINARY)  # Binarize the image
    return Image.fromarray(thresh_image)

# Function to describe the scene in the image using Google Generative AI
def describe_scene(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

    message = HumanMessage(content=[{
        "type": "text", "text": "Describe the content of this image."
    }, {
        "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
    }])

    try:
        response = chat_model.invoke([message])
        return response.content
    except Exception as e:
        return f"Error: {e}"

# Function to detect obstacles in the image using Google Generative AI
def detect_obstacles(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    base64_image = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    message = HumanMessage(content=f"Identify obstacles in this image:\n\n![Image](data:image/jpeg;base64,{base64_image})")

    try:
        response = chat_model.invoke([message])
        return response.content
    except Exception as e:
        return f"Error: {e}"

# Function to resize the image for better display
def resize_image(image, max_width=800, max_height=600):
    width, height = image.size
    
    aspect_ratio = width / height
    
    if width > max_width or height > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        
        image = image.resize((new_width, new_height))
    
    return image

# Streamlit UI for uploading an image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Resize the image before displaying it
    image = resize_image(image)
    
    # Display the image
    st.image(image, caption="Uploaded Image", use_container_width=True)

# Buttons for functionalities
col1, col2, col3 = st.columns(3)
scene_button = col1.button("Describe Scene 🖼️", help="Get a description of the scene in the image", key="scene")
occlusion_button = col2.button("Detect Obstacles 🛑", help="Identify obstacles in the image", key="occlusion")
tts_button = col3.button("Convert Text to Speech 🎙️", help="Convert extracted text to speech", key="tts")

# Main functionality when buttons are clicked
if scene_button:
    with st.spinner("Describing scene..."):
        description = describe_scene(image)
        st.subheader("Scene Description")
        st.write(description)

if occlusion_button:
    with st.spinner("Detecting obstacles..."):
        obstacles = detect_obstacles(image)
        st.subheader("Obstacle Detection")
        st.write(obstacles)

if tts_button:
    with st.spinner("Processing OCR..."):
        processed_image = preprocess_image(image)
        st.image(processed_image, caption="Preprocessed Image for OCR", use_container_width=True)

        raw_text = extract_text_from_image(processed_image)
        cleaned_text = raw_text.strip()

        if cleaned_text:
            st.success("Text extracted successfully!")
            st.subheader("Extracted Text")
            st.write(cleaned_text)

            st.info("Starting Text-to-Speech...")
            text_to_speech(cleaned_text)
        else:
            st.warning("No valid text found in the image. Ensure the image contains clear, readable text.")
