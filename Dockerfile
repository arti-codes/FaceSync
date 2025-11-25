# استفاده از پایتون ۳.۹ (پایدارترین نسخه برای تشخیص چهره)
FROM python:3.9-slim

# جلوگیری از سوالات تعاملی هنگام نصب
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ۱. نصب پیش‌نیازهای سیستمی لینوکس (حیاتی برای dlib)
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgtk-3-dev \
    libboost-all-dev \
    libx11-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# تنظیم پوشه کاری داخل کانتینر
WORKDIR /app

# ۲. کپی کردن و نصب نیازمندی‌های پایتون
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ۳. کپی کردن کل کدهای پروژه به داخل کانتینر
COPY . .

# باز کردن پورت استریم‌لیت
EXPOSE 8501

# ۴. دستور اجرای برنامه
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
