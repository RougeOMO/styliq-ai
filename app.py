import streamlit as st
import cv2
import mediapipe as mp
import google.generativeai as genai
import numpy as np
from PIL import Image
import math
import random
import os 
import re 
import replicate

st.set_page_config(
    page_title="STYLIQ | AI Image Consultant", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {font-family: 'Inter', sans-serif; color: #1a1a1a !important;}
    h1, h2, h3 {font-family: 'Cinzel', serif !important; font-weight: 700 !important; color: #000000 !important;}
    .stApp {background-color: #FFFFFF !important;}
    
    .main-header {
        text-align: center; 
        padding-bottom: 40px;
        border-bottom: 1px solid #F0F0F0;
        margin-bottom: 40px;
    }
    .main-title { font-size: 60px; margin-bottom: 0px; letter-spacing: -1px; }
    .sub-title { font-family: 'Inter'; font-size: 12px; color: #666; letter-spacing: 4px; text-transform: uppercase; margin-top: 10px;}

    .section-header {
        font-family: 'Inter'; 
        font-size: 14px; 
        font-weight: 600; 
        text-transform: uppercase; 
        letter-spacing: 1px;
        margin-bottom: 20px;
        color: #333;
        border-left: 3px solid #000;
        padding-left: 15px;
    }

    .upload-guide {
        background-color: #FAFAFA;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #EEE;
        margin-bottom: 20px;
        font-size: 13px;
        color: #555;
        line-height: 1.6;
    }
    .upload-guide strong { color: #000; font-weight: 600; }

    div.stButton > button[kind="primary"] {
        background-color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        padding: 16px 40px !important;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button[kind="primary"] p {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        font-size: 14px !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    div.stButton > button[kind="primary"]:hover p { color: #000000 !important; }

    .empty-state {
        text-align: center;
        padding: 60px 20px;
        background: #FAFAFA;
        border-radius: 8px;
        border: 1px dashed #DDD;
        color: #999;
    }
    
    [data-testid="stFileUploader"] {padding: 0px;}
    [data-testid="stFileUploader"] section {padding: 30px; background-color: #FFF; border: 1px dashed #CCC;}
    
    </style>
""", unsafe_allow_html=True)

try:
    if "API_KEY" in st.secrets:
        API_KEY = st.secrets["API_KEY"]
        genai.configure(api_key=API_KEY)
    else:
        st.error("🚨 Error: API_KEY missing.")
        st.stop()
        
    if "REPLICATE_API_TOKEN" in st.secrets:
        os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]

    if "SYSTEM_PROMPT" in st.secrets:
        SYSTEM_PROMPT_TEMPLATE = st.secrets["SYSTEM_PROMPT"]
    else:
        st.error("🚨 Error: SYSTEM_PROMPT missing.")
        st.stop()
        
except FileNotFoundError:
    st.error("🚨 Secrets not found.")
    st.stop()

STYLISTS = [
    {"name": "ALEX", "role": "Classic Director", "style": "Timeless Precision", "tone": "Sophisticated, Polite", "avatar": "🏛️"},
    {"name": "JORDAN", "role": "Texture Specialist", "style": "Urban Dynamics", "tone": "Energetic, Modern", "avatar": "⚡"},
    {"name": "CASEY", "role": "Geometric Architect", "style": "Structural Balance", "tone": "Analytical, Sharp", "avatar": "📐"},
    {"name": "TAYLOR", "role": "Natural Consultant", "style": "Organic Flow", "tone": "Warm, Holistic", "avatar": "🌿"}
]

def resize_image(image, max_width=1024):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        image = cv2.resize(image, (max_width, new_h))
    return image

def calculate_distance(p1, p2, w, h):
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def analyze_face(uploaded_file, stylist_persona):
    mp_face_mesh = mp.solutions.face_mesh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, 1)
    image_cv = resize_image(image_cv) 
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    h, w, c = image_cv.shape
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None, None, "⚠️ No face detected."
        landmarks = results.multi_face_landmarks[0].landmark
        face_len = calculate_distance(landmarks[10], landmarks[152], w, h)
        face_width = calculate_distance(landmarks[234], landmarks[454], w, h)
        ratio = face_len / face_width

        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            s_name=stylist_persona['name'],
            s_role=stylist_persona['role'],
            s_style=stylist_persona['style'],
            s_tone=stylist_persona['tone'],
            ratio=f"{ratio:.2f}"
        )
        
        response = model.generate_content(
            [prompt, image_pil], 
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        return image_pil, response.text, None

st.markdown("""
    <div class="main-header">
        <div class="main-title" style="font-family: 'Cinzel', serif;">STYLIQ</div>
        <div class="sub-title">Intelligent Aesthetics</div>
    </div>
""", unsafe_allow_html=True)

if 'current_stylist' not in st.session_state:
    st.session_state['current_stylist'] = None

col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.markdown('<div class="section-header">Step 1: Upload Portrait</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-guide">
        To ensure the most accurate AI analysis, please upload a clear photo:<br><br>
        • <strong>Look Straight:</strong> Face the camera directly.<br>
        • <strong>Even Lighting:</strong> Avoid strong shadows.<br>
        • <strong>No Accessories:</strong> Remove sunglasses or hats.<br>
        <br>
        <span style="font-size: 10px; color: #999;">🔒 Data is processed privately and deleted instantly.</span>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    if uploaded_file:
        st.markdown("---")
        st.image(uploaded_file, caption="Source Image", use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("✨ START ANALYSIS", type="primary"):
            selected_stylist = random.choice(STYLISTS)
            st.session_state['current_stylist'] = selected_stylist
            with st.spinner(f"💎 Analyzing facial geometry..."):
                uploaded_file.seek(0)
                img, report, error = analyze_face(uploaded_file, selected_stylist)
                st.session_state['result'] = (report, error)

with col2:
    st.markdown('<div class="section-header">Step 2: Consultation Report</div>', unsafe_allow_html=True)
    
    stylist = st.session_state['current_stylist']
    
    if stylist and 'result' in st.session_state:
        report, error = st.session_state['result']
        if error:
            st.error(error)
        else:
            s_name = stylist['name']
            s_role = stylist['role'].upper()
            st.markdown(f"""
            <div style="background: #F8F9FA; padding: 15px; border-left: 3px solid #000; margin-bottom: 20px;">
                <div style="font-family: 'Cinzel'; font-weight: 700;">DIRECTOR: {s_name}</div>
                <div style="font-size: 11px; color: #666; letter-spacing: 1px;">{s_role}</div>
            </div>
            """, unsafe_allow_html=True)

            hairstyle_name = "New Hairstyle"
            for line in report.split('\n'):
                if "HAIRSTYLE_NAME:" in line:
                    extracted = line.split(":")[1].strip()
                    if extracted and "[" not in extracted:
                        hairstyle_name = extracted
            if hairstyle_name == "New Hairstyle":
                match = re.search(r"### 3\. Recommendation\s*\n\s*\*\*(.*?)\*\*", report)
                if match:
                    hairstyle_name = match.group(1).strip()

            tab1, tab2, tab3 = st.tabs(["🧬 REPORT", "🖼️ REFERENCES", "✨ AI TRY-ON"])
            
            with tab1:
                clean_report = report.replace(f"HAIRSTYLE_NAME: {hairstyle_name}", "").strip()
                st.markdown(clean_report)
            
            with tab2:
                st.markdown(f"**Recommended Style:** {hairstyle_name}")
                q = urllib.parse.quote(hairstyle_name + " hairstyle reference")
                c1, c2 = st.columns(2)
                with c1: st.link_button("Search Pinterest", f"https://www.pinterest.com/search/pins/?q={q}")
                with c2: st.link_button("Search Google", f"https://www.google.com/search?tbm=isch&q={q}")
            
            with tab3:
                st.info(f"Generating preview for: **{hairstyle_name}**")
                if st.button("Generate Visualization"):
                    if "REPLICATE_API_TOKEN" in st.secrets:
                        try:
                            with st.spinner("Creating your new look..."):
                                with open("temp_upload.jpg", "wb") as f:
                                    uploaded_file.seek(0)
                                    f.write(uploaded_file.read())
                                
                                model_id = "zedge/instantid:ba2d5293be8794a05841a6f6eed81e810340142c3c25fab4838ff2b5d9574420"
                                output = replicate.run(
                                    model_id,
                                    input={
                                        "image": open("temp_upload.jpg", "rb"),
                                        "prompt": f"portrait of a person, {hairstyle_name} hairstyle, photorealistic, 8k, soft lighting, high quality",
                                        "negative_prompt": "bald, distorted face, bad eyes, cartoon, low quality, ugly, messy, painting, drawing",
                                        "ip_adapter_scale": 0.8,
                                        "controlnet_conditioning_scale": 0.8,
                                        "num_inference_steps": 30,
                                        "guidance_scale": 5
                                    }
                                )
                                if output:
                                    st.image(output[0], caption=f"AI Preview: {hairstyle_name}", use_column_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("AI Generation is disabled (Missing Key).")

    else:
        st.markdown("""
        <div class="empty-state">
            <div style="font-size: 40px; margin-bottom: 10px;">🔮</div>
            <div style="font-weight: 600; color: #333;">Awaiting Portrait</div>
            <div style="font-size: 12px; margin-top: 5px;">
                Upload your photo on the left to unlock:<br>
                Face Shape Analysis • Personalized Cut • AI Visuals
            </div>
        </div>
        """, unsafe_allow_html=True)

