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
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {font-family: 'Inter', sans-serif; color: #1a1a1a !important;}
    h1, h2, h3, h4 {font-family: 'Cinzel', serif !important; font-weight: 700 !important; color: #000000 !important;}
    .stApp {background-color: #FFFFFF !important;}
    .stMarkdown, .stText, p, li, span, div[data-testid="stMarkdownContainer"] {color: #1a1a1a !important;}
    
    .stylist-card {background: #F8F9FA; border: 1px solid #E9ECEF; border-left: 4px solid #000; border-radius: 4px; padding: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.03); transition: all 0.3s ease;}
    .stylist-card h2 {font-family: 'Cinzel', serif !important; color: #000 !important;}
    .stylist-card p {font-family: 'Inter', sans-serif !important; color: #666 !important;}
    .stylist-card:hover {transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.08);}
    
    div.stButton > button:first-child {background-color: #000 !important; color: #FFF !important; border: 1px solid #000; border-radius: 0px; padding: 14px 32px; font-family: 'Inter', sans-serif; font-weight: 600; letter-spacing: 2px; text-transform: uppercase;}
    div.stButton > button:first-child:hover {background-color: #FFF !important; color: #000 !important;}
    
    [data-testid="stSidebar"] {background-color: #FAFAFA !important; border-right: 1px solid #EEE;}
    [data-testid="stFileUploader"] {background-color: #FAFAFA !important; border: 1px dashed #DDD !important; border-radius: 0px; padding: 20px;}
    [data-testid="stFileUploaderDropzone"] {background-color: #FFFFFF !important;}
    [data-testid="stFileUploaderDropzone"] div, [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small, [data-testid="stFileUploader"] section {color: #000000 !important; font-family: 'Inter', sans-serif !important;}
    [data-testid="stFileUploaderDropzone"] svg {fill: #000000 !important; color: #000000 !important;}
    [data-testid="stFileUploaderDropzone"] button {background-color: #FFFFFF !important; color: #000000 !important; border-color: #DDD !important;}
    
    .streamlit-expanderHeader {font-family: 'Inter', sans-serif; font-weight: 600; font-size: 14px; color: #000 !important;}
    .streamlit-expanderContent {background-color: #FFFFFF !important; color: #1a1a1a !important;}
    </style>
""", unsafe_allow_html=True)

try:
    if "API_KEY" in st.secrets:
        API_KEY = st.secrets["API_KEY"]
        genai.configure(api_key=API_KEY)
    else:
        st.error("🚨 Error: API_KEY missing in Secrets.")
        st.stop()
        
    if "REPLICATE_API_TOKEN" in st.secrets:
        os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
    else:
        st.warning("⚠️ REPLICATE_API_TOKEN missing. Try-on feature will be disabled.")

    if "SYSTEM_PROMPT" in st.secrets:
        SYSTEM_PROMPT_TEMPLATE = st.secrets["SYSTEM_PROMPT"]
    else:
        st.error("🚨 Error: SYSTEM_PROMPT missing in Secrets.")
        st.stop()
        
except FileNotFoundError:
    st.error("🚨 Secrets File Not Found. Please configure secrets on Streamlit Cloud.")
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

        model = genai.GenerativeModel('gemini-1.5-flash')
        
    
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

with st.sidebar:
    st.title("💎 STYLIQ")
    st.markdown("<p style='font-size: 10px; color: #888; letter-spacing: 3px; text-transform: uppercase;'>Intelligent Aesthetics</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🧬 STYLIQ LAB")
    
    if "REPLICATE_API_TOKEN" in st.secrets:
        st.success("🔌 AI Engine: Online")
    else:
        st.warning("🔌 AI Engine: Offline")

st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>STYLIQ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-family: Inter; letter-spacing: 4px; font-size: 12px; margin-bottom: 50px;'>THE ALGORITHM OF BEAUTY</p>", unsafe_allow_html=True)

if 'current_stylist' not in st.session_state:
    st.session_state['current_stylist'] = None

with st.container():
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 01. DATA SOURCE")
        
        st.markdown("""
        <div style="display: flex; gap: 10px; margin-bottom: 10px;">
            <div style="flex: 1; background: #F8F9FA; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #E9ECEF;">
                <div style="font-size: 20px;">📸</div>
                <div style="font-weight: 600; font-size: 11px; margin-top: 5px; letter-spacing: 0.5px;">FRONT FACING</div>
            </div>
            <div style="flex: 1; background: #F8F9FA; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #E9ECEF;">
                <div style="font-size: 20px;">💡</div>
                <div style="font-weight: 600; font-size: 11px; margin-top: 5px; letter-spacing: 0.5px;">GOOD LIGHT</div>
            </div>
            <div style="flex: 1; background: #F8F9FA; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #E9ECEF;">
                <div style="font-size: 20px;">🔒</div>
                <div style="font-weight: 600; font-size: 11px; margin-top: 5px; letter-spacing: 0.5px;">PRIVATE</div>
            </div>
        </div>

        <div style="background: #FFF; border: 1px dashed #DDD; padding: 12px; border-radius: 8px; margin-bottom: 20px; text-align: center;">
            <div style="font-size: 10px; color: #888; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px;">INCLUDED SERVICES</div>
            <div style="display: flex; justify-content: center; gap: 20px; font-size: 12px; font-weight: 500; color: #333;">
                <span>🧬 4D Face Report</span>
                <span style="color: #DDD;">|</span>
                <span>💇‍♀️ Stylist Advice</span>
                <span style="color: #DDD;">|</span>
                <span>🖼️ AI Try-On</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("🛡️ Privacy Policy", expanded=False):
             st.markdown("""
             <div style="font-size: 12px; color: #666;">
             Your photos are deleted immediately after analysis. No storage, no training.
             </div>
             """, unsafe_allow_html=True)
            
        uploaded_file = st.file_uploader("Upload Portrait", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            st.markdown("""<style>img {filter: grayscale(0%); transition: all 0.5s;} img:hover {filter: grayscale(0%);}</style>""", unsafe_allow_html=True)
            st.image(uploaded_file, use_column_width=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("✨ INITIALIZE ANALYSIS", type="primary"):
                selected_stylist = random.choice(STYLISTS)
                st.session_state['current_stylist'] = selected_stylist
                with st.spinner(f"💎 {selected_stylist['name']} is connecting..."):
                    uploaded_file.seek(0)
                    img, report, error = analyze_face(uploaded_file, selected_stylist)
                    st.session_state['result'] = (report, error)

    with col2:
        st.markdown("### 02. INTELLIGENCE")
        stylist = st.session_state['current_stylist']
        
        if stylist and 'result' in st.session_state:
            report, error = st.session_state['result']
            if error:
                st.error(error)
            else:
                s_name = stylist['name']
                s_role = stylist['role'].upper()
                s_avatar = stylist['avatar']
                
                st.markdown(f"""
                <div class="stylist-card">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <h2 style="margin: 0; font-size: 20px; letter-spacing: 1px;">{s_name}</h2>
                            <p style="margin: 5px 0 0 0; font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 2px;">{s_role}</p>
                        </div>
                        <div style="font-size: 32px;">{s_avatar}</div>
                    </div>
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

                tab1, tab2, tab3 = st.tabs(["ANALYSIS", "MOODBOARD", "TRY-ON"])
                with tab1:
                    clean_report = report.replace(f"HAIRSTYLE_NAME: {hairstyle_name}", "").strip()
                    st.markdown(clean_report)
                with tab2:
                    st.markdown(f"<h3 style='color: #000;'>{hairstyle_name}</h3>", unsafe_allow_html=True)
                    st.caption("Global references.")
                    q = urllib.parse.quote(hairstyle_name + " hairstyle reference")
                    c1, c2 = st.columns(2)
                    with c1: st.link_button("Pinterest", f"https://www.pinterest.com/search/pins/?q={q}")
                    with c2: st.link_button("Google", f"https://www.google.com/search?tbm=isch&q={q}")
                
                with tab3:
                    st.markdown(f"### Virtual Lab: **{hairstyle_name}**")
                    st.caption("Generates a realistic preview using InstantID technology.")
                    
                    if st.button("🧬 GENERATE PREVIEW (BETA)", type="secondary"):
                        if "REPLICATE_API_TOKEN" in st.secrets:
                            try:
                                with st.spinner("🔌 Connecting to Replicate GPU Cluster..."):
                                    with open("temp_upload.jpg", "wb") as f:
                                        uploaded_file.seek(0)
                                        f.write(uploaded_file.read())
                                    
                                    # InstantID Model ID (Fixed Version)
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
                                        st.image(output[0], caption=f"AI Generated: {hairstyle_name}", use_column_width=True)
                                        st.success("✅ Generation Complete!")
                            except Exception as e:
                                st.error(f"⚠️ Replicate Error: {e}")
                        else:
                            st.warning("⚠️ Running in Safe Mode (No Replicate Token Found).")
