import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# ---------- CONFIG ----------
st.set_page_config(page_title="Cleft Lip Detection", layout="wide")

# ---------- FORCE LIGHT THEME + FONT ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

/* FORCE EVERYTHING */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #F6F3EE !important;
    color: #2E3D36 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* REMOVE DARK BLOCKS */
section[data-testid="stSidebar"] {
    background-color: #F6F3EE !important;
}

/* HERO */
.hero {
    text-align:center;
    padding-top:70px;
}
.hero-title {
    font-size:54px;
    font-weight:500;
}
.hero-sub {
    color:#6B7C6F;
    margin-top:10px;
}

/* SECTION */
.section {
    margin-top:80px;
    text-align:center;
}

/* CARD */
.card {
    background:#FFFFFF !important;
    padding:25px;
    border-radius:18px;
    text-align:center;
    box-shadow:0 8px 20px rgba(0,0,0,0.05);
    transition:0.3s;
}
.card:hover {
    transform:translateY(-5px);
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model("model/cleft_model.h5")
IMG_SIZE = 224

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <div class="hero-title">Early Detection. Better Lives.</div>
    <div class="hero-sub">
        AI-powered cleft lip screening for faster and accurate diagnosis.
    </div>
</div>
""", unsafe_allow_html=True)

st.image("assets/hero.png", use_column_width=True)

# ---------- FEATURES ----------
st.markdown("<div class='section'><h2>Core Features</h2></div>", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image("assets/icon_fast.png", width=60)
    st.markdown("### Fast Detection")
    st.caption("Instant AI results")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image("assets/icon_deep_learning.png", width=60)
    st.markdown("### Deep Learning")
    st.caption("Robust CNN architecture")
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image("assets/icon_better_outcomes.png", width=60)
    st.markdown("### Better Outcomes")
    st.caption("Supports early treatment")
    st.markdown("</div>", unsafe_allow_html=True)

c4, c5, c6 = st.columns(3)

with c4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image("assets/icon_high_accuracy.png", width=60)
    st.markdown("### High Accuracy")
    st.caption("Reliable predictions")
    st.markdown("</div>", unsafe_allow_html=True)

with c5:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image("assets/icon_secure.png", width=60)
    st.markdown("### Secure")
    st.caption("Data privacy ensured")
    st.markdown("</div>", unsafe_allow_html=True)

with c6:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.image("assets/icon_trusted.png", width=60)
    st.markdown("### Trusted")
    st.caption("Built with research")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- AWARENESS ----------
st.markdown("<div class='section'><h2>Awareness</h2></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.image("assets/awareness.png")

with col2:
    st.write("""
    Early diagnosis significantly improves treatment success rates.  
    Raising awareness ensures timely intervention and better quality of life.
    """)

# ---------- DETECTION ----------
st.markdown("<div class='section'><h2>Detection</h2></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(img)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

        prediction = model.predict(img)[0][0]

        st.write(f"Confidence: {prediction:.2f}")

        if prediction > 0.8:
            st.error("Cleft Lip Detected")
        else:
            st.success("No Cleft Detected")

with col2:
    st.image("assets/detection.png")

# ---------- TEAM ----------
st.markdown("<div class='section'><h2>Team</h2></div>", unsafe_allow_html=True)

t1, t2, t3 = st.columns(3)

with t1:
    st.image("assets/team_divya.png", width=100)
    st.markdown("**Divya Malhotra**")
    st.caption("AI Development")

with t2:
    st.image("assets/team_divesh.png", width=100)
    st.markdown("**Divesh Singh**")
    st.caption("Model Engineering")

with t3:
    st.image("assets/team_rishit.png", width=100)
    st.markdown("**Rishit Kholiwal**")
    st.caption("Research")

# ---------- FOOTER ----------
st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.markdown("Cleft Lip Detection Platform • 2026")
