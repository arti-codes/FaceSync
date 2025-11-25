import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import math

st.set_page_config(page_title="Face Sync | Arti.codes", page_icon="🧬", layout="centered")

# استایل
st.markdown("""
    <style>
    .stApp {background-color: #050510; color: white;}
    div.stButton > button {background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%); color: white; border: none; padding: 0.6rem 2rem; border-radius: 8px;}
    [data-testid="stFileUploader"] {background-color: #111122; border: 1px dashed #4B5563; border-radius: 10px;}
    h1, h2, h3 {color: #fff !important;}
    </style>
    """, unsafe_allow_html=True)

def process_comparison(image1, image2):
    img1_array = np.array(image1)
    img2_array = np.array(image2)
    
    loc1 = face_recognition.face_locations(img1_array, model="hog")
    loc2 = face_recognition.face_locations(img2_array, model="hog")
    
    if not loc1: return "error", "چهره‌ای در تصویر اول یافت نشد."
    if not loc2: return "error", "چهره‌ای در تصویر دوم یافت نشد."
    
    enc1 = face_recognition.face_encodings(img1_array, known_face_locations=loc1, num_jitters=1)
    enc2 = face_recognition.face_encodings(img2_array, known_face_locations=loc2, num_jitters=1)
    
    if not enc1 or not enc2: return "error", "خطا در استخراج ویژگی‌ها."
    
    dist = face_recognition.face_distance([enc1[0]], enc2[0])[0]
    
    # محاسبه درصد شباهت
    if dist > 0.55:
        score = ((1.0 - dist) / (0.45 * 2.0)) * 100
    else:
        score = (1.0 - (dist / 1.1)) * 100 + 15
        
    return "success", min(score, 99.9)

st.title("سیستم تشخیص چهره Arti")

c1, c2 = st.columns(2)
i1, i2 = None, None

with c1:
    f1 = st.file_uploader("تصویر اول", type=['jpg','png','jpeg'], key="1")
    if f1: i1 = Image.open(f1).convert('RGB'); st.image(i1, use_container_width=True)

with c2:
    f2 = st.file_uploader("تصویر دوم", type=['jpg','png','jpeg'], key="2")
    if f2: i2 = Image.open(f2).convert('RGB'); st.image(i2, use_container_width=True)

if st.button("بررسی تطابق") and i1 and i2:
    with st.spinner('در حال پردازش...'):
        status, res = process_comparison(i1, i2)
        if status == "error": st.error(res)
        else:
            col = "#4ade80" if res > 60 else "#f87171"
            msg = "تایید شد ✅" if res > 60 else "رد شد ❌"
            st.markdown(f"<div style='text-align:center; border:2px solid {col}; padding:20px; border-radius:10px;'><h1 style='color:{col}; margin:0;'>{res:.1f}%</h1><h3>{msg}</h3></div>", unsafe_allow_html=True)
