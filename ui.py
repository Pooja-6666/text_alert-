import streamlit as st
import requests
def explain_action(state, stress, energy):
    if stress >= 4:
        return "High stress detected → grounding helps stabilize your mind."
    if energy <= 2:
        return "Low energy → rest is recommended."
    if state in ["calm", "focused"]:
        return "Good mental state → best time for deep work."
    return "Taking a pause helps you reset."

st.set_page_config(page_title="ArvyaX", page_icon="🌿", layout="centered")

# 🎨 Background color function
def set_bg_color(state):
    color_map = {
        "happy": "#1f7a1f",
        "stressed": "#7a1f1f",
        "calm": "#1f3d7a",
        "sad": "#4b3f72"
    }
    color = color_map.get(state, "#0e1117")

    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {color};
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)


# 🎨 Default styling
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
    }
    .stTextArea textarea {
        background-color: #1e1e1e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🌿 ArvyaX Emotional AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Understand your emotions. Take smarter actions.</p>", unsafe_allow_html=True)

st.divider()

# Input section
st.subheader("📝 Your Input")

text = st.text_area("How are you feeling?", placeholder="Write your thoughts...")

col1, col2, col3 = st.columns(3)

with col1:
    stress = st.slider("😣 Stress", 1, 5, 3)

with col2:
    energy = st.slider("⚡ Energy", 1, 5, 3)

with col3:
    sleep = st.slider("😴 Sleep", 0, 10, 6)

st.divider()

# Button
if st.button("🔍 Analyze", use_container_width=True):

    with st.spinner("Analyzing your emotions..."):
        response = requests.post(
            "https://text-alert.onrender.com/predict",
            json={
                "text": text,
                "stress": stress,
                "energy": energy,
                "sleep": sleep
            }
        )

    # ✅ FIXED INDENTATION
    if response.status_code == 200:
        result = response.json()

        # 🎨 Apply background
        set_bg_color(result["state"])

        # 🎯 Emotion-based UI
        if result["state"] == "happy":
            st.balloons()
            st.success("You're doing amazing today! 🚀")

        elif result["state"] == "stressed":
            st.warning("Pause. Breathe. Reset. 🌬️")

        elif result["state"] == "calm":
            st.success("You're in a great state ✨")

        elif result["state"] == "sad":
            st.info("It's okay to slow down 💙")

        # 📊 Results
        st.success("Analysis Complete ✅")

        st.subheader("🧠 Results")
        

        st.markdown(f"""
        <div style='padding:25px; border-radius:15px; 
        background: linear-gradient(135deg, #1e1e1e, #2c2c2c);
        box-shadow: 0 0 15px rgba(0,0,0,0.5);'>

        <h3>🧠 Emotion: {result['state']}</h3>
        <h3>🔥 Intensity: {result['intensity']}</h3>
        <h3>🎯 Action: {result['action']}</h3>
        <h3>⏰ When: {result['timing']}</h3>

        </div>
        """, unsafe_allow_html=True)
        st.progress(result["confidence"])
        st.caption(f"Confidence: {round(result['confidence'] * 100, 2)}%")

        st.info(f"💬 {result['message']}")
        st.caption(f"🧠 Why: {explain_action(result['state'], stress, energy)}")

    else:
        st.error("API not working ❌")