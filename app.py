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
import urllib.parse
import io
from pathlib import Path

# --- 1. 强制页面重置 ---
st.set_page_config(
    page_title="STYLIQ 4.0 | Ultimate", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. 样式加载 ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif; color: #1a1a1a !important;}
    h1, h2, h3 {font-family: 'Cinzel', serif !important; font-weight: 700 !important; color: #000000 !important;}
    .stApp {background-color: #FFFFFF !important;}
    
    .main-header {text-align: center; padding-bottom: 40px; border-bottom: 1px solid #F0F0F0; margin-bottom: 40px;}
    .main-title {font-size: 60px; margin-bottom: 0px; letter-spacing: -1px;}
    .sub-title {font-family: 'Inter'; font-size: 12px; color: #666; letter-spacing: 4px; text-transform: uppercase; margin-top: 10px;}
    
    .section-header {font-family: 'Inter'; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; color: #333; border-left: 3px solid #000; padding-left: 15px;}
    .upload-guide {background-color: #FAFAFA; padding: 20px; border-radius: 8px; border: 1px solid #EEE; margin-bottom: 20px; font-size: 13px; color: #555; line-height: 1.6;}
    .upload-guide strong {color: #000; font-weight: 600;}
    
    div.stButton > button[kind="primary"] {background-color: #000 !important; border: 1px solid #000; padding: 16px 40px !important; width: 100%; transition: all 0.3s ease;}
    div.stButton > button[kind="primary"] p {color: #FFF !important; font-family: 'Inter'; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; font-size: 14px;}
    div.stButton > button[kind="primary"]:hover {background-color: #FFF !important; color: #000 !important;}
    div.stButton > button[kind="primary"]:hover p {color: #000 !important;}
    
    .empty-state {text-align: center; padding: 60px 20px; background: #FAFAFA; border-radius: 8px; border: 1px dashed #DDD; color: #999;}
    [data-testid="stFileUploader"] section {padding: 30px; background-color: #FFF; border: 1px dashed #CCC;}
    </style>
""", unsafe_allow_html=True)

# --- 3. 密钥检查 ---
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
    st.error(f"🚨 Secrets Error: {e}")
    st.stop()

# --- 4. 核心逻辑 (完全不依赖 uploaded_file) ---
STYLISTS = [
    {"name": "ALEX", "role": "Classic Director", "style": "Timeless Precision", "tone": "Sophisticated", "avatar": "🏛️"},
    {"name": "JORDAN", "role": "Texture Specialist", "style": "Urban Dynamics", "tone": "Modern", "avatar": "⚡"},
    {"name": "CASEY", "role": "Geometric Architect", "style": "Structural Balance", "tone": "Sharp", "avatar": "📐"},
    {"name": "TAYLOR", "role": "Natural Consultant", "style": "Organic Flow", "tone": "Holistic", "avatar": "🌿"}
]

def calculate_ratio(image_cv):
    try:
        h, w, c = image_cv.shape
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0].landmark
            # 简单计算：脸长(10-152) / 脸宽(234-454)
            y1, y2 = landmarks[10].y * h, landmarks[152].y * h
            x1, x2 = landmarks[234].x * w, landmarks[454].x * w
            length = math.sqrt((y2 - y1)**2)
            width = math.sqrt((x2 - x1)**2)
            return length / width
    except:
        return 1.5 # 默认值

def analyze_logic(img_data, stylist):
    # 解码图片
    file_bytes = np.asarray(bytearray(img_data), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, 1)
    if image_cv is None: return None, "Image Decode Error"
    
    # 计算比例
    ratio = calculate_ratio(image_cv)
    if ratio is None: return None, "No Face Detected"
    
    # 准备 PIL
    image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    # 调用 Gemini
    try:
        # 优先尝试 Flash-Lite
        model = genai.GenerativeModel('gemini-2.5-flash') 
        response = model.generate_content(
            [SYSTEM_PROMPT_TEMPLATE.format(
                s_name=stylist['name'],
                s_role=stylist['role'],
                s_style=stylist['style'],
                s_tone=stylist['tone'],
                ratio=f"{ratio:.2f}"
            ), image_pil],
            generation_config=genai.types.GenerationConfig(temperature=0.2)
        )
        return response.text, None
    except Exception as e:
        return None, f"AI Error: {str(e)}"

# --- 5. UI 构建 ---
st.markdown('<div class="main-header"><div class="main-title">STYLIQ 4.0</div><div class="sub-title">Intelligent Aesthetics</div></div>', unsafe_allow_html=True)

# 初始化 Session State (改名以防冲突)
if 'user_img_data' not in st.session_state: st.session_state['user_img_data'] = None
if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None
if 'active_stylist' not in st.session_state: st.session_state['active_stylist'] = None

col1, col2 = st.columns([4, 6], gap="large")

# 左侧：上传
with col1:
    st.markdown('<div class="section-header">Step 1: Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-guide">To ensure accurate analysis, please upload a clear, front-facing photo with good lighting.</div>', unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
    
    if uploaded:
        st.image(uploaded, use_container_width=True)
        
        if st.button("✨ START ANALYSIS", type="primary"):
            # 1. 立即保存数据到 Session
            bytes_data = uploaded.getvalue()
            st.session_state['user_img_data'] = bytes_data
            
            # 2. 随机选人
            stylist = random.choice(STYLISTS)
            st.session_state['active_stylist'] = stylist
            
            # 3. 分析
            with st.spinner("Analyzing..."):
                report, err = analyze_logic(bytes_data, stylist)
                st.session_state['analysis_result'] = (report, err)

# 右侧：结果
with col2:
    st.markdown('<div class="section-header">Step 2: Results</div>', unsafe_allow_html=True)
    
    res = st.session_state['analysis_result']
    stylist = st.session_state['active_stylist']
    
    if res and stylist:
        report_text, error_text = res
        
        if error_text:
            st.error(error_text)
        else:
            # 显示造型师
            st.markdown(f"""
            <div style="background:#F8F9FA; padding:15px; border-left:3px solid #000; margin-bottom:20px;">
                <b>DIRECTOR: {stylist['name']}</b><br><span style="font-size:12px; color:#666;">{stylist['role']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # 提取发型名
            style_name = "New Look"
            match = re.search(r"HAIRSTYLE_NAME:\s*(.*)", report_text)
            if match: style_name = match.group(1).strip()
            
            # Tabs
            t1, t2, t3 = st.tabs(["REPORT", "REF", "TRY-ON"])
            
            with t1:
                st.markdown(report_text.replace(f"HAIRSTYLE_NAME: {style_name}", ""))
                
            with t2:
                q = urllib.parse.quote(style_name + " hairstyle")
                st.link_button("Search Pinterest", f"https://www.pinterest.com/search/pins/?q={q}")
                
            with t3:
                st.info(f"Generating: **{style_name}**")
                
                # --- 这里的逻辑最关键：完全脱离 uploaded_file ---
                if st.session_state['user_img_data'] is None:
                    st.warning("⚠️ Session expired. Please upload again.")
                else:
                    if st.button("Generate Visualization"):
                        try:
                            with st.spinner("Rendering..."):
                                # 1. 使用绝对路径保存临时文件
                                temp_path = Path("temp_face.jpg").resolve()
                                with open(temp_path, "wb") as f:
                                    f.write(st.session_state['user_img_data'])
                                
                                # 2. 显式打开文件句柄
                                if not temp_path.exists():
                                    st.error("Temp file creation failed.")
                                else:
                                    # 使用 with open 确保文件正确传递
                                    with open(temp_path, "rb") as img_file:
                                        output = replicate.run(
                                            "zedge/instantid:ba2d5293be8794a05841a6f6eed81e810340142c3c25fab4838ff2b5d9574420",
                                            input={
                                                "image": img_file,
                                                "prompt": f"portrait of a person, {style_name} hairstyle, photorealistic, 8k",
                                                "negative_prompt": "bald, distorted, bad eyes, low quality, illustration",
                                                "ip_adapter_scale": 0.8,
                                                "controlnet_conditioning_scale": 0.8
                                            }
                                        )
                                    
                                    if output:
                                        st.image(output[0], caption="AI Generated Result", use_container_width=True)
                                        
                        except Exception as e:
                            st.error(f"Generation Error: {e}")

    else:
        st.markdown('<div class="empty-state">🔮<br>Awaiting Portrait</div>', unsafe_allow_html=True)

