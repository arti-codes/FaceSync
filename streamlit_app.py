import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import math

# --- 1. تنظیمات صفحه و استایل (CSS) ---
st.set_page_config(
    page_title="Face Scan - Arti.codes",
    page_icon="🧬",
    layout="centered"
)

# استایل اختصاصی برای هماهنگی با تم تیره و نئونی سایت شما
st.markdown("""
    <style>
    .stApp {
        background-color: #050510;
        color: white;
    }
    /* دکمه‌ها با گرادینت */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: transform 0.2s;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.02);
        border: 1px solid #fff;
    }
    /* باکس‌های آپلود */
    [data-testid="stFileUploader"] {
        background-color: #111122;
        border: 1px dashed #4B5563;
        border-radius: 10px;
        padding: 20px;
    }
    h1, h2, h3 { color: #ffffff !important; font-family: sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. توابع منطقی (Logic) - برگرفته از کد اصلی شما ---

def face_distance_to_conf(face_distance, face_match_threshold=0.55):
    if face_distance > face_match_threshold:
        range_val = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range_val * 2.0)
        return linear_val * 100
    else:
        range_val = face_match_threshold
        linear_val = 1.0 - (face_distance / (range_val * 2.0))
        similarity = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        bonus = 15 * (1 - (similarity / 100)) 
        final_score = similarity + bonus
        return min(final_score, 99.9)

def process_comparison(image1, image2):
    # تبدیل تصاویر PIL به فرمت مورد نیاز face_recognition
    img1_array = np.array(image1)
    img2_array = np.array(image2)

    status_container = st.empty() # برای نمایش وضعیت لحظه‌ای

    try:
        # مرحله 1: یافتن چهره‌ها
        status_container.info("🔍 در حال یافتن چهره در تصویر اول...")
        locations1 = face_recognition.face_locations(img1_array, model="hog")
        if not locations1:
            return "error", "❌ چهره‌ای در تصویر اول پیدا نشد."

        status_container.info("🧠 در حال استخراج ویژگی‌های چهره اول...")
        encodings1 = face_recognition.face_encodings(img1_array, known_face_locations=locations1, num_jitters=1, model="large")

        status_container.info("🔍 در حال یافتن چهره در تصویر دوم...")
        locations2 = face_recognition.face_locations(img2_array, model="hog")
        if not locations2:
            return "error", "❌ چهره‌ای در تصویر دوم پیدا نشد."

        status_container.info("🧠 در حال استخراج ویژگی‌های چهره دوم...")
        encodings2 = face_recognition.face_encodings(img2_array, known_face_locations=locations2, num_jitters=1, model="large")

        if not encodings1 or not encodings2:
            return "error", "❌ خطا در استخراج ویژگی‌های چهره."

        # مقایسه
        status_container.info("⚡ در حال مقایسه نهایی...")
        encoding1 = encodings1[0]
        encoding2 = encodings2[0]

        face_dist = face_recognition.face_distance([encoding1], encoding2)[0]
        similarity = face_distance_to_conf(face_dist)
        
        status_container.empty() # پاک کردن پیام‌های وضعیت
        return "success", similarity

    except Exception as e:
        return "error", f"خطای ناشناخته: {str(e)}"

# --- 3. رابط کاربری (UI) ---

st.title("Face Scan App Test")
st.markdown("---")

col1, col2 = st.columns(2)

image1 = None
image2 = None

with col1:
    st.subheader("تصویر اول")
    file1 = st.file_uploader("انتخاب تصویر اول", type=['jpg', 'png', 'jpeg'], key="1")
    if file1:
        image1 = Image.open(file1).convert('RGB')
        st.image(image1, use_container_width=True)

with col2:
    st.subheader("تصویر دوم")
    file2 = st.file_uploader("انتخاب تصویر دوم", type=['jpg', 'png', 'jpeg'], key="2")
    if file2:
        image2 = Image.open(file2).convert('RGB')
        st.image(image2, use_container_width=True)

st.markdown("---")

# دکمه اجرا
if st.button("شروع مقایسه دقیق"):
    if image1 and image2:
        with st.spinner('در حال پردازش...'):
            status, result = process_comparison(image1, image2)
        
        if status == "error":
            st.error(result)
        else:
            similarity = result
            
            # منطق نمایش رنگ و پیام طبق کد اصلی شما
            if similarity > 90:
                msg_color = "#4ade80" # Green
                msg_text = "تطابق بسیار بالا (تایید شده) ✅"
            elif similarity > 60:
                msg_color = "#fbbf24" # Orange
                msg_text = "تطابق متوسط ⚠️"
            else:
                msg_color = "#f87171" # Red
                msg_text = "عدم تطابق ❌"

            st.markdown(f"""
            <div style="background-color: #1e1e2e; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {msg_color};">
                <h1 style="color: {msg_color} !important; margin: 0;">{similarity:.1f}%</h1>
                <h3 style="margin-top: 10px;">{msg_text}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(int(similarity))
    else:
        st.warning("لطفاً هر دو تصویر را بارگذاری کنید.")