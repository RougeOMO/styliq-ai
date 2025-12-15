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
import urllib.parse
import io
from pathlib import Path

st.set_page_config(
    page_title="STYLIQ | Cloud V31", 
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
    
    .main-header {text-align: center; padding-bottom: 20px; border-bottom: 1px solid #F0F0F0; margin-bottom: 40px;}
    .main-title {font-size: 60px; margin-bottom: 0px; letter-spacing: -1px;}
    .sub-title {font-family: 'Inter'; font-size: 12px; color: #666; letter-spacing: 4px; text-transform: uppercase; margin-top: 10px;}
    
    .section-header {font-family: 'Inter'; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; color: #333; border-left: 3px solid #000; padding-left: 15px;}
    .upload-guide {background-color: #FAFAFA; padding: 20px; border-radius: 8px; border: 1px solid #EEE; margin-bottom: 20px; font-size: 13px; color: #555; line-height: 1.6;}
    
    div.stButton > button[kind="primary"] {background-color: #000 !important; border: 1px solid #000; border-radius: 0px; padding: 16px 40px; width: 100%; transition: all 0.3s ease;}
    div.stButton > button[kind="primary"] p {color: #FFF !important; font-family: 'Inter'; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; font-size: 14px;}
    div.stButton > button[kind="primary"]:hover {background-color: #FFF !important; color: #000 !important;}
    div.stButton > button[kind="primary"]:hover p {color: #000 !important;}
    
    .empty-state {text-align: center; padding: 60px 20px; background: #FAFAFA; border-radius: 8px; border: 1px dashed #DDD; color: #999;}
    [data-testid="stFileUploader"] {padding: 0px;}
    [data-testid="stFileUploader"] section {padding: 30px; background-color: #FFF; border: 1px dashed #CCC;}
    </style>
""", unsafe_allow_html=True)

try:
    if "API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["API_KEY"])
    else:
        st.error("🚨 Missing API_KEY")
        st.stop()
        
    if "REPLICATE_API_TOKEN" in st.secrets:
        os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
    
    if "SYSTEM_PROMPT" in st.secrets:
        SYSTEM_PROMPT_TEMPLATE = st.secrets["SYSTEM_PROMPT"]
    else:
        st.error("🚨 Missing SYSTEM_PROMPT")
        st.stop()
except Exception as e:
    st.error(f"🚨 Secrets Config Error: {e}")
    st.stop()

STYLISTS = [
    {"name": "ALEX", "role": "Classic Director", "style": "Timeless", "tone": "Sophisticated", "avatar": "🏛️"},
    {"name": "JORDAN", "role": "Texture Specialist", "style": "Urban", "tone": "Modern", "avatar": "⚡"},
    {"name": "CASEY", "role": "Geometric Architect", "style": "Structural", "tone": "Sharp", "avatar": "📐"},
    {"name": "TAYLOR", "role": "Natural Consultant", "style": "Organic", "tone": "Holistic", "avatar": "🌿"}
]

def analyze_logic(image_bytes, stylist):
    try:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, 1)
        if image_cv is None: return None, "Image Decode Failed"
        
        h, w, c = image_cv.shape
        mp_face = mp.solutions.face_mesh
        with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks: return None, "No Face Detected"
            landmarks = results.multi_face_landmarks[0].landmark
            ratio = (landmarks[152].y - landmarks[10].y) * h / ((landmarks[454].x - landmarks[234].x) * w)

        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content([SYSTEM_PROMPT_TEMPLATE.format(s_name=stylist['name'], s_role=stylist['role'], s_style=stylist['style'], s_tone=stylist['tone'], ratio=f"{ratio:.2f}"), image_pil])
            return response.text, None
        except Exception as e:
            return None, f"AI Error: {str(e)}"
    except Exception as e:
        return None, f"Analysis Error: {str(e)}"

st.markdown("""
    <div class="main-header">
        <div class="main-title">STYLIQ</div>
        <div class="sub-title">Intelligent Aesthetics</div>
    </div>
""", unsafe_allow_html=True)

if 'user_img_bytes' not in st.session_state: st.session_state['user_img_bytes'] = None
if 'analysis_data' not in st.session_state: st.session_state['analysis_data'] = None
if 'active_stylist' not in st.session_state: st.session_state['active_stylist'] = None

col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.markdown('<div class="section-header">Step 1: Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-guide">Please upload a front-facing photo. Data is processed securely.</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if uploaded:
        st.image(uploaded, use_container_width=True)
        
        if st.button("✨ START ANALYSIS", type="primary"):
            st.session_state['user_img_bytes'] = uploaded.getvalue()
            
            stylist = random.choice(STYLISTS)
            st.session_state['active_stylist'] = stylist
            
            with st.spinner("Analyzing..."):
                rep, err = analyze_logic(st.session_state['user_img_bytes'], stylist)
                st.session_state['analysis_data'] = (rep, err)

with col2:
    st.markdown('<div class="section-header">Step 2: Results</div>', unsafe_allow_html=True)
    
    res = st.session_state['analysis_data']
    stylist = st.session_state['active_stylist']
    
    if res and stylist:
        rep_text, err_text = res
        
        if err_text:
            st.error(err_text)
        else:
            style_name = "New Look"
            match = re.search(r"HAIRSTYLE_NAME:\s*(.*)", rep_text)
            if match: style_name = match.group(1).strip()
            
            st.markdown(f"""
            <div style="background:#F8F9FA; padding:15px; border-left:3px solid #000; margin-bottom:20px;">
                <b>DIRECTOR: {stylist['name']}</b><br><span style="font-size:12px; color:#666;">{stylist['role']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            t1, t2, t3 = st.tabs(["REPORT", "REF", "TRY-ON"])
            
            with t1:
                st.markdown(rep_text.replace(f"HAIRSTYLE_NAME: {style_name}", ""))
                
            with t2:
                q = urllib.parse.quote(style_name + " hairstyle")
                st.link_button("Search Pinterest", f"https://www.pinterest.com/search/pins/?q={q}")
                
            with t3:
                st.info(f"Visualizing: **{style_name}**")
                
                if st.session_state['user_img_bytes'] is None:
                    st.warning("⚠️ Please re-upload your photo.")
                else:
                    if st.button("Generate Visualization"):
                        try:
                            with st.spinner("Rendering (this takes ~10s)..."):
                                temp_file = Path("temp_process.jpg")
                                temp_file.write_bytes(st.session_state['user_img_bytes'])
                                
                                if temp_file.exists():
                                    with open(temp_file, "rb") as f_handle:
                                        output = replicate.run(
                                            "zedge/instantid:ba2d5293be8794a05841a6f6eed81e810340142c3c25fab4838ff2b5d9574420",
                                            input={
                                                "image": f_handle,
                                                "prompt": f"portrait of a person, {style_name} hairstyle, photorealistic, 8k",
                                                "negative_prompt": "bald, distorted, bad eyes, low quality, illustration",
                                                "ip_adapter_scale": 0.8,
                                                "controlnet_conditioning_scale": 0.8,
                                                "num_inference_steps": 30,
                                                "guidance_scale": 5
                                            }
                                        )
                                    if output:
                                        st.image(output[0], caption="AI Prediction", use_container_width=True)
                        except Exception as e:
                            st.error(f"Generation Error: {e}")
    else:
        st.markdown('<div class="empty-state">🔮<br>Awaiting Portrait</div>', unsafe_allow_html=True)
