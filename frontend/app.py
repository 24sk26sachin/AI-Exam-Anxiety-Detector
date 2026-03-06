import streamlit as st
import requests
import json

# Page Configuration
st.set_page_config(
    page_title="Exam Anxiety Detector",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .low-anxiety {
        background-color: #d4edda;
        color: #155724;
        border-left: 5px solid #28a745;
    }
    .moderate-anxiety {
        background-color: #fff3cd;
        color: #856404;
        border-left: 5px solid #ffc107;
    }
    .high-anxiety {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 5px solid #dc3545;
    }
    .tip-box {
        background-color: white;
        color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🧠 AI Exam Anxiety Detector")
st.markdown("""
Welcome to the **AI-Based Exam Anxiety Detector**. This tool helps you understand your current emotional state regarding upcoming exams. 
Simply share your thoughts or feelings below, and our AI will provide feedback and tips to help you manage your stress.
""")

st.divider()

# Input area
user_input = st.text_area("How are you feeling about your exams?", placeholder="e.g., I'm feeling a bit nervous but I've studied hard...", height=150)

if st.button("Analyze Anxiety Level"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing your thoughts..."):
            try:
                # Call FastAPI Backend
                backend_url = "http://localhost:8000/predict"
                response = requests.post(backend_url, json={"text": user_input})
                
                if response.status_code == 200:
                    data = response.json()
                    level = data["anxiety_level"]
                    confidence = data["confidence"]

                    # Display Result
                    if "Low" in level:
                        css_class = "low-anxiety"
                        emoji = "✅"
                        tips = [
                            "Keep up the good work! You seem to have a healthy handle on your exams.",
                            "Continue your current study routine and ensure you get enough sleep.",
                            "Try a 5-minute mindfulness session to maintain this calm state."
                        ]
                    elif "Moderate" in level:
                        css_class = "moderate-anxiety"
                        emoji = "⚠️"
                        tips = [
                            "It's normal to feel some pressure. Take short breaks every 45 minutes.",
                            "Focus on active recall and testing yourself to build confidence.",
                            "Practice deep breathing exercises: inhale for 4, hold for 4, exhale for 4."
                        ]
                    else:
                        css_class = "high-anxiety"
                        emoji = "🚨"
                        tips = [
                            "Take a step back. Your well-being is more important than any single exam.",
                            "Talk to a mentor, teacher, or counselor about how you're feeling.",
                            "Try the '5-4-3-2-1' grounding technique to calm your nervous system.",
                            "Break your study material into very small, manageable chunks."
                        ]

                    st.markdown(f"""
                    <div class="result-card {css_class}">
                        <h3>{emoji} Result: {level}</h3>
                        <p>Confidence: {confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.subheader("💡 Tips for You:")
                    for tip in tips:
                        st.markdown(f"""<div class="tip-box">{tip}</div>""", unsafe_allow_html=True)

                else:
                    st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.divider()
st.info("Disclaimer: This tool is for support purposes only and is not a clinical diagnostic tool.")
