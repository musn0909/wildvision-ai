import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import pandas as pd
import time

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="WildVision AI",
    page_icon="🐾",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

html, body, [class*="css"] {

    font-family: 'Segoe UI', sans-serif;
}

.stApp {

    background: linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #111827
    );

    color: white;
}

.block-container {

    max-width: 1200px;

    padding-top: 2rem;
}

.hero-card {

    background: linear-gradient(
        to right,
        rgba(37,99,235,0.2),
        rgba(124,58,237,0.2)
    );

    border: 1px solid rgba(255,255,255,0.08);

    padding: 35px;

    border-radius: 25px;

    margin-bottom: 30px;

    backdrop-filter: blur(14px);
}

.glass-card {

    background: rgba(255,255,255,0.05);

    border: 1px solid rgba(255,255,255,0.08);

    padding: 25px;

    border-radius: 22px;

    backdrop-filter: blur(12px);

    margin-bottom: 25px;
}

.stButton > button {

    width: 100%;

    height: 3.5em;

    border-radius: 16px;

    border: none;

    background: linear-gradient(
        to right,
        #2563eb,
        #7c3aed
    );

    color: white;

    font-size: 18px;

    font-weight: bold;
}

[data-testid="stImage"] img {

    border-radius: 20px;

    box-shadow: 0px 0px 30px rgba(0,0,0,0.4);
}

</style>
""", unsafe_allow_html=True)

# ---------------- CLASSES ----------------

classes = [
    "Dog",
    "Horse",
    "Elephant",
    "Butterfly",
    "Chicken",
    "Cat",
    "Cow",
    "Sheep",
    "Spider",
    "Squirrel"
]

# ---------------- ANIMAL FACTS ----------------

animal_facts = {

    "Dog":
    "Dogs are highly intelligent and loyal animals.",

    "Horse":
    "Horses can sleep standing up.",

    "Elephant":
    "Elephants are the largest land animals.",

    "Butterfly":
    "Butterflies taste using their feet.",

    "Chicken":
    "Chickens recognize over 100 faces.",

    "Cat":
    "Cats sleep almost 70% of their lives.",

    "Cow":
    "Cows have excellent social memory.",

    "Sheep":
    "Sheep remember faces for years.",

    "Spider":
    "Spiders live on every continent except Antarctica.",

    "Squirrel":
    "Squirrels accidentally plant trees."
}

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "models/animal_classifier.h5"
    )

model = load_model()

# ---------------- HERO SECTION ----------------

st.markdown("""

<div class="hero-card">

<h1 style="font-size:60px;">
🐾 WildVision AI
</h1>

<h3 style="color:#cbd5e1;">
Deep Learning Animal Recognition System
</h3>

<p style="color:#94a3b8; font-size:18px;">

Upload an animal image and let the AI model
analyze and predict the species using
computer vision and transfer learning.

</p>

</div>

""", unsafe_allow_html=True)

# ---------------- TRAINING CATEGORIES ----------------

st.markdown("""

<div class="glass-card">

<h3>🧠 AI Training Categories</h3>

<p style="font-size:17px; color:#cbd5e1;">

WildVision AI is trained using a custom
deep learning dataset containing the following species:

<b>
Dog • Horse • Elephant • Butterfly • Chicken •
Cat • Cow • Sheep • Spider • Squirrel
</b>

<br><br>

The model achieves highest accuracy when
predicting supported animal categories.

</p>

</div>

""", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(

    "Upload Animal Image",

    type=["jpg", "jpeg", "png"]
)

# ---------------- MAIN SECTION ----------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    # ---------------- IMAGE PANEL ----------------

    with col1:

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    # ---------------- AI PANEL ----------------

    with col2:

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        st.subheader("🚀 AI Analysis")

        predict_button = st.button(
            "Start Prediction"
        )

        if predict_button:

            # ---------------- LOADING UI ----------------

            progress = st.progress(0)

            status = st.empty()

            loading_steps = [

                "Loading neural network...",

                "Analyzing image...",

                "Extracting features...",

                "Running inference...",

                "Generating predictions..."
            ]

            for i in range(100):

                progress.progress(i + 1)

                if i < 20:
                    status.info(loading_steps[0])

                elif i < 40:
                    status.info(loading_steps[1])

                elif i < 60:
                    status.info(loading_steps[2])

                elif i < 80:
                    status.info(loading_steps[3])

                else:
                    status.info(loading_steps[4])

                time.sleep(0.02)

            # ---------------- PREPROCESS IMAGE ----------------

            img = image.resize((224, 224))

            img_array = np.array(img)

            img_array = img_array / 255.0

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            # ---------------- PREDICTION ----------------

            predictions = model.predict(img_array)

            confidence = np.max(predictions) * 100

            predicted_index = np.argmax(predictions)

            predicted_class = classes[predicted_index]

            # ---------------- UNKNOWN DETECTION ----------------

            if confidence < 60:

                st.error(
                    "Unknown Animal / Unsupported Image"
                )

                st.caption(
                    "Predictions outside trained categories may produce lower confidence scores."
                )

            else:

                st.success(
                    f"{predicted_class} Detected"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.caption(
                    "Predictions outside trained categories may produce lower confidence scores."
                )

                # ---------------- FACTS ----------------

                st.subheader("📚 Animal Facts")

                st.write(
                    animal_facts[predicted_class]
                )

            # ---------------- TOP PREDICTIONS ----------------

            st.subheader("📊 Top Predictions")

            top_indices = np.argsort(
                predictions[0]
            )[-3:][::-1]

            top_animals = [
                classes[i]
                for i in top_indices
            ]

            top_scores = [
                float(predictions[0][i] * 100)
                for i in top_indices
            ]

            chart_df = pd.DataFrame({

                "Animal": top_animals,

                "Confidence": top_scores
            })

            fig = px.bar(

                chart_df,

                x="Animal",

                y="Confidence",

                text="Confidence",

                color="Confidence"
            )

            fig.update_traces(
                texttemplate='%{text:.1f}%',
                textposition='outside'
            )

            fig.update_layout(

                height=350,

                paper_bgcolor='rgba(0,0,0,0)',

                plot_bgcolor='rgba(0,0,0,0)',

                font_color='white',

                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )