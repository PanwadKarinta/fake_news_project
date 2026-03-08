import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# โหลดโมเดล
@st.cache_resource
def load_models():
    model = joblib.load("best_random_forest_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

# ค่า Accuracy
@st.cache_resource
def load_accuracy():
    return joblib.load("model_accuracy.pkl")

model, vectorizer = load_models()
MODEL_ACCURACY = round(load_accuracy() * 100, 2)

# 🎨 ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)

# 📊 Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3524/3524335.png", width=100)
    st.markdown("## 📌 About This App")
    st.info("""
    ระบบตรวจสอบข่าวปลอมด้วย AI ที่ใช้ Machine Learning 
    ในการวิเคราะห์ข้อความข่าว
    """)
    
    st.markdown("### 📊 Model Performance")
    st.metric("Accuracy", f"{MODEL_ACCURACY}%", "High Performance")
    
    st.markdown("### ⚙️ How It Works")
    st.markdown("""
    1. **Input**: ใส่ข้อความข่าว
    2. **Processing**: วิเคราะห์ด้วย TF-IDF + Random Forest
    3. **Output**: คาดการณ์พร้อมความมั่นใจ
    """)
    
    st.markdown("### ⚠️ คำเตือน")
    st.warning("""
    - ระบบนี้เป็นเครื่องมือช่วยตัดสินใจเบื้องต้น
    - ควรตรวจสอบข้อมูลจากแหล่งที่เชื่อถือได้เพิ่มเติม
    - ไม่ควรใช้เป็นข้อมูลอ้างอิงเดียว
    """)

# 🎯 Header
st.markdown('<h1 class="main-header">🔍 Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown("---")

# 📝 Layout หลัก
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 ใส่ข้อความข่าวที่ต้องการตรวจสอบ")
    text = st.text_area(
        "News Text",
        height=250,
        placeholder="วางข้อความข่าวที่ต้องการตรวจสอบที่นี่...",
        label_visibility="collapsed"
    )
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        predict_btn = st.button("🔍 ตรวจสอบความน่าเชื่อถือ", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ ล้างข้อความ", use_container_width=True):
            st.rerun()

with col2:
    st.markdown("### 💡 ตัวอย่างข่าว")
    
    example_real = st.expander("✅ ตัวอย่างข่าวจริง")
    with example_real:
        st.write("ON BOARD A U.S. MILITARY AIRCRAFT (Reuters) - The U.S. Air Force may intensify its strikes in Afghanistan and expand training of the Afghan air force following President Donald Trump s decision to forge ahead with the 16-year-old war, its top general told Reuters on Tuesday. Air Force Chief of Staff General David Goldfein said, however, he was still examining the matter, as the U.S. military s top brass had only begun the process of translating Trump s war strategy into action. Asked whether the Air Force would dedicate more assets to Afghanistan")
    
    example_fake = st.expander("❌ ตัวอย่างข่าวปลอม")
    with example_fake:
        st.write("BREAKING: Aliens landed in Bangkok! Government hiding the truth! Share before they delete this!!!")

# 🎯 การทำนาย
if predict_btn:
    if text.strip() == "":
        st.warning("⚠️ กรุณาใส่ข้อความก่อนกดตรวจสอบ")
    else:
        with st.spinner("🔄 กำลังวิเคราะห์..."):
            # ทำนายผล
            transformed = vectorizer.transform([text])
            prediction = model.predict(transformed)[0]
            probability = model.predict_proba(transformed)
            confidence = round(float(probability.max()) * 100, 2)
            
            # แสดงผล
            st.markdown("---")
            st.markdown("## 📊 ผลการวิเคราะห์")
            
            result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
            
            with result_col2:
                if prediction == "REAL":
                    st.success("### ✅ ข่าวน่าเชื่อถือ (REAL)")
                    st.balloons()
                else:
                    st.error("### ❌ ข่าวน่าสงสัย (FAKE)")
                
                # Progress bar
                st.markdown(f"**ความมั่นใจ: {confidence}%**")
                st.progress(int(confidence))
                
                # แสดงความน่าจะเป็นทั้งสองแบบ
                st.markdown("### 📈 รายละเอียดการวิเคราะห์")
                prob_df = pd.DataFrame({
                    'ประเภท': ['FAKE', 'REAL'],
                    'ความน่าจะเป็น (%)': [
                        round(float(probability[0][0]) * 100, 2),
                        round(float(probability[0][1]) * 100, 2)
                    ]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # คำแนะนำ
                st.markdown("### 💭 คำแนะนำ")
                if confidence > 80:
                    st.info("🎯 ระบบมีความมั่นใจสูง แต่ควรตรวจสอบแหล่งที่มาเพิ่มเติม")
                elif confidence > 60:
                    st.warning("⚠️ ระบบมีความมั่นใจปานกลาง ควรตรวจสอบจากหลายแหล่ง")
                else:
                    st.error("❗ ระบบมีความมั่นใจต่ำ ควรตรวจสอบอย่างละเอียด")
            
            # บันทึกประวัติ (optional)
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'result': prediction,
                'confidence': confidence
            })

# 📜 ประวัติการตรวจสอบ
if 'history' in st.session_state and len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown("### 📜 ประวัติการตรวจสอบ")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎓 Developed for Educational Purposes | Khon Kaen University</p>
        <p style='font-size: 0.8rem;'>⚠️ This is a learning tool. Always verify information from reliable sources.</p>
    </div>
    """,
    unsafe_allow_html=True
)