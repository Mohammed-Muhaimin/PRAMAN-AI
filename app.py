import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time
import io
import hashlib  # For SHA-256 hashing
import json     # For the ledger
import os       # For checking if files exist
import google.generativeai as genai # For GPT-OSS insights

# --- 1. Constants and Model Parameters ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
MODEL_PATH = 'my_change_detection_model.h5'
LEDGER_FILE = 'ledger.json' 

# --- API KEY (Hardcoded as requested) ---
# !!! --- SECURITY WARNING --- !!!
# DO NOT share this file or upload it to GitHub.
GEMINI_API_KEY = "AIzaSyAp2o7qDzuQlffFmRroW1ouA6PmiKRduDo"
# --- END WARNING ---

# --- 2. Hashing & Ledger Functions (Same as before) ---
def calculate_hash(image_file):
    image_file.seek(0)
    file_bytes = image_file.read()
    sha256_hash = hashlib.sha256()
    sha256_hash.update(file_bytes)
    image_file.seek(0)
    return sha256_hash.hexdigest()

def load_ledger():
    if not os.path.exists(LEDGER_FILE):
        return {}
    try:
        with open(LEDGER_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_to_ledger(filename, file_hash):
    ledger = load_ledger()
    ledger[filename] = file_hash
    with open(LEDGER_FILE, 'w') as f:
        json.dump(ledger, f, indent=4)
    st.success(f"File '{filename}' registered to ledger with hash: {file_hash[:10]}...")

def verify_hash(image_file):
    if image_file is None:
        return "No file uploaded."
    filename = image_file.name
    current_hash = calculate_hash(image_file)
    ledger = load_ledger()
    if filename not in ledger:
        return "UNVERIFIED (File not in ledger)"
    official_hash = ledger[filename]
    if current_hash == official_hash:
        return "VERIFIED"
    else:
        return "TAMPERED"

# --- 3. Keras Model Functions (Same as before) ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1. - score

def iou(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred > 0.5, tf.bool)
    intersection = tf.logical_and(y_true, y_pred)
    union = tf.logical_or(y_true, y_pred)
    return (tf.reduce_sum(tf.cast(intersection, tf.float32)) + 1e-6) / (tf.reduce_sum(tf.cast(union, tf.float32)) + 1e-6)

@st.cache_resource
def load_tf_model(model_path):
    try:
        model = load_model(
            model_path,
            custom_objects={'dice_loss': dice_loss, 'iou': iou}
        )
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}. Did you place it in the same folder as app.py?\nError: {e}")
        return None

# --- 4. App Core Functions (Same as before) ---
def preprocess_image(image_file):
    # This function expects an uploaded file object, which we'll keep
    img = Image.open(image_file).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def run_prediction(model, img_a, img_b):
    input_a = preprocess_image(img_a) 
    input_b = preprocess_image(img_b)
    pred_mask = model.predict((input_a, input_b))
    pred_mask_squeezed = pred_mask[0]
    change_pixels = np.sum(pred_mask_squeezed > 0.5)
    total_pixels = pred_mask_squeezed.size
    change_percentage = (change_pixels / total_pixels) * 100
    return pred_mask_squeezed, change_percentage

# --- 5. Flexible Multimodal Gemini AI Insight Function (Same as before) ---

# We still need TWO system prompts: one for images-only, one for fusion.
IMAGE_ONLY_PROMPT = """
You are an expert geospatial analyst for ISRO. Your task is to analyze two satellite images, 'Image T1 (Before)' and 'Image T2 (After)', and provide a concise, professional report.

Your report must have three sections:
1.  **Visual Change Summary:** A brief, 1-2 sentence description of the *most significant changes* visible between the two images.
2.  **Change Classification:** Classify the primary type of change. Your options are: 'Urban Growth', 'New Construction', 'Deforestation', 'Revegetation/Reforestation', 'Agricultural Change', 'Flood Event', 'No Significant Change', or 'Other'.
3.  **Actionable Insight:** Based on the change, provide a 1-2 sentence insight. For example, 'This area shows active deforestation; recommend monitoring' or 'This appears to be seasonal revegetation; no alert needed.'
"""

MULTIMODAL_PROMPT = """
You are an expert geospatial analyst and multimodal fusion engine for ISRO, acting as the GPT-OSS. 
Your task is to analyze a set of multimodal data:
1.  'Image T1 (Before)' - A satellite image.
2.  'Sensor Data T1 (Before)' - A JSON/text file with telemetry for the first image.
3.  'Image T2 (After)' - A satellite image of the same location.
4.  'Sensor Data T2 (After)' - A JSON/text file with telemetry for the second image.

Your task is to FUSE this information. Use the sensor data (like vegetation_index, soil_moisture, or telemetry_log) to explain the *context* of the visual changes you see in the images.

Your report must have three sections:
1.  **Visual Change Summary:** A 1-2 sentence description of the *visual changes* in the images.
2.  **Sensor Data Correlation:** A 1-2 sentence explanation of how the sensor data *supports or explains* the visual change (e.g., "The visual revegetation is confirmed by the sensor data, which shows a 500% increase in the vegetation_index.").
3.  **Actionable Insight & Risk Assessment:** Based on all data, provide a 1-2 sentence insight. Classify the event (e.g., 'Revegetation', 'Deforestation', 'Flood Event', 'Construction') and state a risk level (e.g., 'No threat' or 'High risk, monitor for flooding').
"""

# Configure the API key ONCE at the start
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API. Is the key valid? Error: {e}")

# This function is already flexible and accepts strings
@st.cache_data
def get_ai_insight(img_a, img_b, change_percent, sensor_a_txt=None, sensor_b_txt=None):
    """
    Calls the Gemini API. If sensor_a_txt and sensor_b_txt are provided
    (and are not empty strings), it performs a full multimodal fusion. 
    Otherwise, it performs a visual-only analysis.
    """
    try:
        img_a_pil = Image.open(img_a).convert('RGB')
        img_b_pil = Image.open(img_b).convert('RGB')
        
        # --- Check if sensor text is meaningful (not None or empty) ---
        if sensor_a_txt and sensor_b_txt:
            # --- 1. MULTIMODAL FUSION PATH ---
            system_prompt = MULTIMODAL_PROMPT
            user_prompt = f"""
            Analyze the following data packet. My U-Net change detection model has already calculated that {change_percent:.2f}% of the pixels have changed.
            
            ---
            Data Packet T1 (Before):
            - Image T1 (Before): [Image 1 is attached]
            - Sensor Data T1 (Before):
            ```
            {sensor_a_txt}
            ```
            ---
            Data Packet T2 (After):
            - Image T2 (After): [Image 2 is attached]
            - Sensor Data T2 (After):
            ```
            {sensor_b_txt}
            ```
            ---
            
            Please provide your fused analysis report.
            """
            report_type = "Multimodal Fusion (Images + Sensor Data)"
            
        else:
            # --- 2. IMAGE-ONLY PATH ---
            system_prompt = IMAGE_ONLY_PROMPT
            user_prompt = f"""
            Analyze these two images. My U-Net model detected a {change_percent:.2f}% pixel change.
            Image T1 (Before) is first, Image T2 (After) is second.
            """
            report_type = "Visual-Only Analysis (Images)"
        # --- END NEW LOGIC ---

        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-preview-09-2025',
            system_instruction=system_prompt
        )
        
        response = model.generate_content([user_prompt, img_a_pil, img_b_pil])
        
        return response.text, report_type # Return both the text and the type of report
    
    except Exception as e:
        return f"Error during AI analysis: {e}", "Error"

# --- 6. Main Streamlit App UI ---

# Set page title and icon
st.set_page_config(page_title="Secure EO Analysis", layout="wide")

# Load the Keras model
model = load_tf_model(MODEL_PATH)

# Main Title
st.title("🛰️ PRAMAN AI : Analyse & Safeguard ISRO EO data 🌍")
st.markdown("A demonstration of secure, multimodal vision capabilities for interpreting ISRO's Earth Observation data.")

# Initialize session state 
if 'verified_A' not in st.session_state:
    st.session_state.verified_A = False
if 'verified_B' not in st.session_state:
    st.session_state.verified_B = False
if 'change_mask' not in st.session_state:
    st.session_state.change_mask = None
if 'change_percent' not in st.session_state:
    st.session_state.change_percent = 0.0
if 'before_file' not in st.session_state:
    st.session_state.before_file = None
if 'after_file' not in st.session_state:
    st.session_state.after_file = None
# --- NEW: Changed from file object to text string ---
if 'sensor_a_txt' not in st.session_state:
    st.session_state.sensor_a_txt = ""
if 'sensor_b_txt' not in st.session_state:
    st.session_state.sensor_b_txt = ""


tab1, tab2, tab3 = st.tabs(["[1] Analysis & Verification", "[2] Security Ledger", "[3] GPT-OSS Insight"])

# --- Tab 1: The Main Model Interface (NOW WITH TEXT_AREA) ---
with tab1:
    st.header("Upload Multimodal Data Packet")
    st.info("Upload your verified images. Sensor data is optional but provides deeper insights.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Packet T1 (Before)")
        before_file = st.file_uploader("Upload 'Before' Image (T1)", type=['png', 'jpg', 'jpeg'], key="uploader_A")
        
        # --- NEW: Changed to st.text_area ---
        sensor_a_txt_input = st.text_area("Paste 'Before' Sensor Data (Optional)", height=150, key="uploader_SA")

        if before_file:
            st.session_state.before_file = before_file
            status = verify_hash(before_file)
            if status == "VERIFIED":
                st.success(f"✅ Image Status: {status}")
                st.session_state.verified_A = True
            elif status == "TAMPERED":
                st.error(f"❌ Image Status: {status}")
                st.session_state.verified_A = False
            else:
                st.warning(f"⚠️ Image Status: {status}")
                st.session_state.verified_A = False
        
        # --- NEW: Save text to session state ---
        if sensor_a_txt_input:
            st.session_state.sensor_a_txt = sensor_a_txt_input
            st.success("✅ Sensor data T1 loaded.")
        else:
            st.session_state.sensor_a_txt = "" # Ensure it's empty if user deletes text

    with col2:
        st.subheader("Data Packet T2 (After)")
        after_file = st.file_uploader("Upload 'After' Image (T2)", type=['png', 'jpg', 'jpeg'], key="uploader_B")
        
        # --- NEW: Changed to st.text_area ---
        sensor_b_txt_input = st.text_area("Paste 'After' Sensor Data (Optional)", height=150, key="uploader_SB")
        
        if after_file:
            st.session_state.after_file = after_file
            status = verify_hash(after_file)
            if status == "VERIFIED":
                st.success(f"✅ Image Status: {status}")
                st.session_state.verified_B = True
            elif status == "TAMPERED":
                st.error(f"❌ Image Status: {status}")
                st.session_state.verified_B = False
            else:
                st.warning(f"⚠️ Image Status: {status}")
                st.session_state.verified_B = False
        
        # --- NEW: Save text to session state ---
        if sensor_b_txt_input:
            st.session_state.sensor_b_txt = sensor_b_txt_input
            st.success("✅ Sensor data T2 loaded.")
        else:
            st.session_state.sensor_b_txt = "" # Ensure it's empty if user deletes text

    st.divider()

    # U-Net analysis only needs the two images
    can_run_unet = st.session_state.verified_A and st.session_state.verified_B and model
    
    if st.button("Run U-Net Change Analysis", use_container_width=True, type="primary", disabled=not can_run_unet):
        with st.spinner("U-Net Model is analyzing visual changes..."):
            mask, percent = run_prediction(model, before_file, after_file)
            st.session_state.change_mask = mask
            st.session_state.change_percent = percent
            st.success("U-Net Analysis Complete! You can now generate an AI Insight in Tab 3.")
    
    # Display U-Net results
    if st.session_state.change_mask is not None:
        st.subheader("U-Net Visual Analysis Results:")
        col_img1, col_img2, col_mask = st.columns(3)
        col_img1.image(before_file, caption="Before (T1)", use_container_width=True)
        col_img2.image(after_file, caption="After (T2)", use_container_width=True)
        # Safely render the predicted mask (convert numpy arrays to a grayscale PIL image)
        try:
            mask_arr = st.session_state.change_mask
            if isinstance(mask_arr, np.ndarray):
                if mask_arr.ndim == 3 and mask_arr.shape[-1] == 1:
                    mask_arr = np.squeeze(mask_arr, axis=-1)
                mask_min = mask_arr.min()
                mask_range = np.ptp(mask_arr)
                mask_norm = (mask_arr - mask_min) / (mask_range + 1e-8)
                mask_uint8 = (mask_norm * 255).astype('uint8')
                mask_img = Image.fromarray(mask_uint8, mode='L')
                col_mask.image(mask_img, caption="Predicted Change Mask (White = Change)", use_container_width=True)
            else:
                col_mask.image(mask_arr, caption="Predicted Change Mask (White = Change)", use_container_width=True)
        except Exception as e:
            col_mask.error(f"Error displaying mask: {e}")
        st.info(f"The U-Net model detected visual change in **{st.session_state.change_percent:.2f}%** of the pixels.")

# --- Tab 2: Security & Ledger Management (Same as before) ---
with tab2:
    st.header("Image Ledger Management (Simulated Blockchain)")
    st.markdown("This is where you register 'official' images to the ledger. This simulates ISRO securely publishing their data.")
    st.subheader("Register a New Image")
    register_file = st.file_uploader("Upload an image to register", type=['png', 'jpg', 'jpeg'], key="register_uploader")

    if st.button("Register Image to Ledger", use_container_width=True):
        if register_file:
            file_hash = calculate_hash(register_file)
            save_to_ledger(register_file.name, file_hash)
        else:
            st.warning("Please upload a file to register.")
    st.divider()
    with st.expander("View Current On-Chain Ledger (`ledger.json`)"):
        st.json(load_ledger())

# --- Tab 3: GPT-OSS Insight (NOW FLEXIBLE) ---
with tab3:
    st.header("Actionable Insights from GPT-OSS (Live AI)")
    st.markdown("This tab uses a live model to fuse all available data (images and optional sensor text) to generate a final report.")
    
    # The button is active as long as the U-Net has run.
    can_run_gpt = st.session_state.change_mask is not None
    
    if not can_run_gpt:
        st.warning("Please run a [VERIFIED] analysis in Tab 1 to generate insights.")
    
    if st.button("Generate AI Insight", use_container_width=True, type="primary", disabled=not can_run_gpt):
        
        # --- NEW: Get text directly from session state ---
        sensor_a_txt = st.session_state.sensor_a_txt
        sensor_b_txt = st.session_state.sensor_b_txt
        
        spinner_text = "🛰️ Calling GPT-OSS.. The AI is performing a **visual-only analysis**..."
        
        # Check if optional sensor text was provided (i.e., strings are not empty)
        if sensor_a_txt and sensor_b_txt:
            spinner_text = "🛰️ Calling GPT-OSS... The AI is **fusing** images and sensor data..."

        with st.spinner(spinner_text):
            # Get the *actual* AI report
            report_text, report_type = get_ai_insight(
                st.session_state.before_file, 
                st.session_state.after_file, 
                st.session_state.change_percent,
                sensor_a_txt, # This will be an empty string if not provided
                sensor_b_txt  # This will be an empty string if not provided
            )
            
            # Format the final report
            final_report = f"""
            **Analysis Target:** User-uploaded multimodal data packet.
            **Data Integrity:** [VERIFIED - Image Hashes Matched Ledger]
            **Analysis Type:** {report_type}
            **U-Net Visual Analysis:** Detected {st.session_state.change_percent:.2f}% pixel change.
            
            ---
            
            **[GPT-OSS Multimodal Fusion Report]**
            
            {report_text}
            """
            st.text_area("Live AI-Generated Report:", final_report, height=500)
    
    elif can_run_gpt:
        st.info("All data is ready. Click the button above to generate the final multimodal report.")