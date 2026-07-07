import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configurations for a wide, modern look
st.set_page_config(
    page_title="Student Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a creative and clean layout
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="main-title">🎓 Student Performance Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the student behavioral and academic metrics in the sidebar to forecast outcomes.</div>', unsafe_allow_html=True)

# Load the trained pickle model safely
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ 'model.pkl' not found. Please rename your pickle file to 'model.pkl' and place it in the root directory.")
    st.stop()

# Sidebar Inputs for features
st.sidebar.header("📥 Student Metrics Input")

study_hours = st.sidebar.number_input("Study Hours", min_value=0, max_value=24, value=5)
attendance = st.sidebar.slider("Attendance Rate (%)", min_value=0, max_value=100, value=85)
sleep_hours = st.sidebar.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
internet_usage = st.sidebar.selectbox("Internet Usage (0=No, 1=Yes)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
assignments_completed = st.sidebar.number_input("Assignments Completed", min_value=0, max_value=50, value=10)
previous_score = st.sidebar.slider("Previous Score", min_value=0, max_value=100, value=75)
exam_score = st.sidebar.slider("Current Exam Score Mock", min_value=0, max_value=100, value=70)

# Create input structure matching training features
input_data = pd.DataFrame([{
    'study_hours': study_hours,
    'attendance': attendance,
    'sleep_hours': sleep_hours,
    'internet_usage': internet_usage,
    'assignments_completed': assignments_completed,
    'previous_score': previous_score,
    'exam_score': exam_score
}])

# Center layout for results
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.write("### 📊 Metrics Summary View")
    st.dataframe(input_data, hide_index=True)
    
    if st.button("🚀 Analyze Performance", use_container_width=True):
        # Predict using model
        prediction_num = model.predict(input_data)[0]
        
        # Grab probabilities if model supports it
        try:
            probabilities = model.predict_proba(input_data)[0]
            confidence = probabilities[prediction_num] * 100
        except:
            confidence = None

        st.markdown("---")
        st.write("### 🎯 Classification Result")
        
        # Convert numerical prediction to categorical
        if prediction_num == 1:
            category_result = "Pass"
            st.balloons()
            st.success(f"🎉 **Status:** {category_result}")
        else:
            category_result = "Fail / Risk of Failure"
            st.sidebar.warning("Attention Required")
            st.error(f"⚠️ **Status:** {category_result}")
            
        if confidence:
            st.info(f"🔍 **Model Confidence Level:** {confidence:.2f}%")
