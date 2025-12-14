import streamlit as st
import cv2
import mediapipe as mp
import google.generativeai as genai
import numpy as np
from PIL import Image
import math
import urllib.parse
import random
import time 
import os 
import re 

st.set_page_config(
    page_title="STYLIQ | AI Image Consultant", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif; 
    }
    
    .stApp {
        background-color: #FFFFFF !important;
    }
    
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, li, span, div[data-testid="stMarkdownContainer"] {
        color: #1a1a1a !important;
    }
    
    .stylist-card {
        background: #F8F9FA; 
        border: 1px solid #E9ECEF; 
        border-left: 4px solid #000;
        border-radius: 4px; 
        padding: 25px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03); 
        transition: all 0.3s ease;
    }
    .stylist-card h2, .stylist-card p {
        color: #1a1a1a !important;
    }
    .stylist-card:hover {transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0,0,0,0.08);}
    
    div.stButton > button:first-child {
        background-color: #000 !important; 
        color: #FFF !important;
        border: 1px solid #000;
        border-radius: 0px; 
        padding: 14px 32px; 
        font-family: 'Inter'; 
        font-weight: 600; 
        letter-spacing: 2px; 
        text-transform: uppercase;
    }
    div.stButton > button:first-child:hover {
        background-color: #FFF !important; 
        color: #000 !important;
    }
    
    [data-testid="stSidebar"] {background-color: #FAFAFA; border-right: 1px solid #EEE;}
    [data-testid="stFileUploader"] {background-color: #FAFAFA; border: 1px dashed #DDD; border-radius: 0px; padding: 30px;}
    
    .streamlit-expanderHeader {
        font-family: 'Inter'; 
        font-weight: 600; 
        font-size: 14px;
        color: #000 !important;
    }
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
        color: #1a1a1a !important;
    }
    </style>
""", unsafe_allow_html=True)

try:
    if "API_KEY" in st.secrets:
        API_KEY = st.secrets["API_KEY"]
    else:
        st.error("🚨 STYLIQ SYSTEM ERROR: API Key missing in Secrets.")
        st.stop()
except FileNotFoundError:
    st.error("🚨 Local Configuration Missing. Please create .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=API_KEY)

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
            return None, None, "⚠️ No face detected. Please try a clearer photo."
            
        landmarks = results.multi_face_landmarks[0].landmark
        face_len = calculate_distance(landmarks[10], landmarks[152], w, h)
        face_width = calculate_distance(landmarks[234], landmarks[454], w, h)
        ratio = face_len / face_width

        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        s_name = stylist_persona['name']
        s_role = stylist_persona['role']
        s_style = stylist_persona['style']
        s_tone = stylist_persona['tone']

        try:
            raw_prompt = st.secrets["SYSTEM_PROMPT"]
        except KeyError:
            return None, None, "🚨 Error: SYSTEM_PROMPT missing in Secrets."

        prompt = raw_prompt.format(
            s_name=s_name,
            s_role=s_role,
            s_style=s_style,
            s_tone=s_tone,
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
    st.info("Virtual Try-On Module")
    replicate_api = st.text_input("API Key (Optional)", type="password", placeholder="Enter key to unlock...")
    if not replicate_api:
        st.caption("Running in **Simulation Mode** (Free)")

st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>STYLIQ</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; font-family: Inter; letter-spacing: 4px; font-size: 12px; margin-bottom: 50px;'>THE ALGORITHM OF BEAUTY</p>", unsafe_allow_html=True)

if 'current_stylist' not in st.session_state:
    st.session_state['current_stylist'] = None

with st.container():
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### 01. DATA SOURCE")
        
        with st.expander("📸 **READ ME: How to get the best results**", expanded=True):
            st.markdown("""
            **For the most accurate analysis, please upload:**
            * ✅ **Front-facing**: Look directly at the camera.
            * ✅ **Good Lighting**: Ensure your face is evenly lit.
            * ✅ **No Obstructions**: Remove sunglasses.
            
            **What you will receive:**
            * 🧬 **4D Face Report**: Face Shape, Hair Type, Color Analysis, and Vibe.
            * 💇‍♀️ **Expert Advice**: A personalized hairstyle recommendation.
            * 🖼️ **Visual Preview**: A simulation of the look.
            """)
            
        uploaded_file = st.file_uploader("Upload Portrait", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            st.markdown("""<style>img {filter: grayscale(0%); transition: all 0.5s;} img:hover {filter: grayscale(0%);}</style>""", unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("✨ INITIALIZE ANALYSIS", type="primary", use_container_width=True):
                selected_stylist = random.choice(STYLISTS)
                st.session_state['current_stylist'] = selected_stylist
                with st.spinner(f"💎 {selected_stylist['name']} is connecting with your vibe..."):
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
                s_style = stylist['style']
                
                st.markdown(f"""
                <div class="stylist-card">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <h2 style="margin: 0; font-size: 20px; letter-spacing: 1px;">{s_name}</h2>
                            <p style="margin: 5px 0 0 0; font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 2px;">{s_role}</p>
                        </div>
                        <div style="font-size: 32px;">{s_avatar}</div>
                    </div>
                    <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #EEE;">
                        <p style="font-family: 'Inter'; font-size: 14px; color: #333;">"{s_style}"</p>
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
                    
                    with c1: st.link_button("Pinterest", f"https://www.pinterest.com/search/pins/?q={q}", use_container_width=True)
                    with c2: st.link_button("Google", f"https://www.google.com/search?tbm=isch&q={q}", use_container_width=True)
                
                with tab3:
                    st.markdown(f"### Virtual Lab: **{hairstyle_name}**")
                    st.caption("Generates a preview using STYLIQ Diffusion Engine.")
                    
                    if st.button("🧬 GENERATE PREVIEW", type="secondary", use_container_width=True):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        steps = [
                            "🔌 Connecting to Neural Engine...",
                            "📐 Mapping Facial Geometry...",
                            "🎨 Synthesizing Texture...",
                            "✨ Rendering Final Polish..."
                        ]
                        
                        for i in range(100):
                            time.sleep(0.02) 
                            progress_bar.progress(i + 1)
                            if i < 25: status_text.text(steps[0])
                            elif i < 50: status_text.text(steps[1])
                            elif i < 75: status_text.text(steps[2])
                            else: status_text.text(steps[3])
                        
                        status_text.text("✅ Computation Complete")
                        
                        if replicate_api:
                            st.warning("API Key detected! (Real logic placeholder)")
                        else:
                            if os.path.exists("demo.jpg"):
                                st.image("demo.jpg", caption=f"STYLIQ Preview: {hairstyle_name}", use_container_width=True)
                            else:
                                st.warning("⚠️ 'demo.jpg' not found. Please add a demo image.")
                                st.image("https://images.unsplash.com/photo-1560250097-0b93528c311a?w=800&q=80", caption="Fallback", use_container_width=True)
                            
                            st.info("ℹ️ Running in **Simulation Mode**.")
