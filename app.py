import streamlit as st
import cv2
import mediapipe as mp
import google.generativeai as genai
import numpy as np
from PIL import Image
import math
import random
import os
import replicate
import tempfile  # ✅ 新增：用于创建带后缀的标准临时文件

st.set_page_config(
    page_title="STYLIQ V34", 
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
    div.stButton > button[kind="primary"] {background-color: #000 !important; border: 1px solid #000; border-radius: 0px; padding: 16px 40px; width: 100%; transition: all 0.3s ease;}
    div.stButton > button[kind="primary"] p {color: #FFF !important; font-family: 'Inter'; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; font-size: 14px;}
    div.stButton > button[kind="primary"]:hover {background-color: #FFF !important; color: #000 !important;}
    div.stButton > button[kind="primary"]:hover p {color: #000 !important;}
    </style>
""", unsafe_allow_html=True)

try:
    if "API_KEY" in st.secrets: genai.configure(api_key=st.secrets["API_KEY"])
    if "REPLICATE_API_TOKEN" in st.secrets: os.environ["REPLICATE_API_TOKEN"] = st.secrets["REPLICATE_API_TOKEN"]
    if "SYSTEM_PROMPT" in st.secrets: SYSTEM_PROMPT_TEMPLATE = st.secrets["SYSTEM_PROMPT"]
except Exception:
    st.error("🚨 Secrets Config Error")
    st.stop()

STYLISTS = [{"name": "ALEX", "role": "Director", "style": "Timeless", "tone": "Sophisticated", "avatar": "🏛️"}]

def analyze_logic(image_bytes, stylist):
    try:
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, 1)
        if image_cv is None: return None, "Decode Error"
        
        # 简单计算比例
        h, w, c = image_cv.shape
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks: return None, "No Face Detected"
            landmarks = results.multi_face_landmarks[0].landmark
            ratio = (landmarks[152].y - landmarks[10].y) * h / ((landmarks[454].x - landmarks[234].x) * w)

        image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        
        try:
            model = genai.GenerativeModel('gemini-2.5-flash-lite')
            response = model.generate_content([SYSTEM_PROMPT_TEMPLATE.format(s_name=stylist['name'], s_role=stylist['role'], s_style=stylist['style'], s_tone=stylist['tone'], ratio=f"{ratio:.2f}"), image_pil])
            return response.text, None
        except Exception as e:
            return None, f"AI Error: {str(e)}"
    except Exception as e:
        return None, f"Sys Error: {str(e)}"

st.markdown('<div style="text-align: center; margin-bottom: 40px;"><h1 style="font-family:Cinzel; font-size:60px;">STYLIQ</h1></div>', unsafe_allow_html=True)

if 'user_img_bytes' not in st.session_state: st.session_state['user_img_bytes'] = None
if 'analysis_data' not in st.session_state: st.session_state['analysis_data'] = None

col1, col2 = st.columns([4, 6], gap="large")

with col1:
    st.markdown("### Step 1: Upload")
    uploaded = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if uploaded:
        st.image(uploaded, use_container_width=True)
        if st.button("✨ START ANALYSIS", type="primary"):
            st.session_state['user_img_bytes'] = uploaded.getvalue()
            with st.spinner("Analyzing..."):
                rep, err = analyze_logic(st.session_state['user_img_bytes'], STYLISTS[0])
                st.session_state['analysis_data'] = (rep, err)

with col2:
    st.markdown("### Step 2: Results")
    res = st.session_state['analysis_data']
    
    if res:
        rep_text, err_text = res
        if err_text:
            st.error(err_text)
        else:
            style_name = "New Look"
            import re
            match = re.search(r"HAIRSTYLE_NAME:\s*(.*)", rep_text)
            if match: style_name = match.group(1).strip()
            
            st.info(f"Recommended: **{style_name}**")
            st.markdown(rep_text.replace(f"HAIRSTYLE_NAME: {style_name}", ""), unsafe_allow_html=True)
            
            if st.session_state['user_img_bytes']:
                if st.button("Generate Visualization"):
                    try:
                        with st.spinner("Rendering on Cloud..."):
                            # ✅ 核心修复：使用 tempfile 显式创建 .jpg 文件
                            # 这确保了 Replicate 能正确识别文件类型
                            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                                temp_file.write(st.session_state['user_img_bytes'])
                                temp_path = temp_file.name
                            
                            # 重新打开并发送
                            with open(temp_path, "rb") as file_handle:
                                output = replicate.run(
                                    "zedge/instantid:ba2d5293be8794a05841a6f6eed81e810340142c3c25fab4838ff2b5d9574420",
                                    input={
                                        "image": file_handle,
                                        "prompt": f"portrait of a person, {style_name} hairstyle, photorealistic, 8k",
                                        "negative_prompt": "bald, distorted, bad eyes, low quality, illustration",
                                        "ip_adapter_scale": 0.8,
                                        "controlnet_conditioning_scale": 0.8
                                    }
                                )
                            
                            if output:
                                st.image(output[0], caption="AI Prediction", use_container_width=True)
                                
                    except Exception as e:
                        st.error(f"Error: {e}")
