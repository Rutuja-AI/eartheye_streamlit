# [START OF FILE]


import streamlit as st
import numpy as np
import os, json, uuid, traceback
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

st.set_page_config(page_title="EarthEye Classifier", layout="centered")

# --- Optional: Enhanced Title & Info ---
st.title("🌍 EarthEye")
st.markdown("### AI-powered Earth Feature Recognition using Satellite Imagery 🛰️")

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

def preprocess_image_file(file):
    try:
        img = image.load_img(file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array)
    except Exception as e:
        st.error(f"Error preprocessing: {e}")
        return None

# --- Load Class Names ---
CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
               'River', 'SeaLake']
class_indices_path = os.path.join('models', 'class_indices.json')
if os.path.exists(class_indices_path):
    try:
        with open(class_indices_path, 'r') as f:
            indices = json.load(f)
        CLASS_NAMES = [None] * len(indices)
        for k, v in indices.items():
            CLASS_NAMES[v] = k
    except:
        pass

def get_feature_info(label):
    descs = {
        'annualcrop': ("Agricultural land for seasonal crops.", ['Farming', 'Food']),
        'forest': ("Wooded area with biodiversity.", ['Nature', 'Trees']),
        'herbaceousvegetation': ("Grassland and meadows.", ['Grass', 'Natural']),
        'highway': ("Transportation routes.", ['Urban', 'Road']),
        'industrial': ("Factories and warehouses.", ['Industry']),
        'pasture': ("Grazing areas for livestock.", ['Agriculture']),
        'permanentcrop': ("Orchards, plantations.", ['Perennial', 'Farming']),
        'residential': ("Housing areas.", ['Urban', 'Living']),
        'river': ("Freshwater rivers and streams.", ['Water']),
        'sealake': ("Large water bodies.", ['Marine', 'Ocean'])
    }
    l = label.lower()
    d = descs.get(l, (f"Predicted feature: {label}", [label]))
    return {'description': d[0], 'features': d[1]}

# --- Load Keras Model ---
KERAS_MODEL_PATH = os.path.join("models", "earth_classifier")  # No .keras
model = None
if os.path.exists(KERAS_MODEL_PATH):
    model = load_model(KERAS_MODEL_PATH)
else:
    st.error("Keras model not found at models/earth_classifier")

st.write("Upload a satellite image to classify its land cover type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif"])

if uploaded_file is not None:
    if not allowed_file(uploaded_file.name):
        st.error("Invalid file type. Please upload a PNG, JPG, JPEG, or GIF.")
    else:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")
        img = preprocess_image_file(uploaded_file)
        if img is not None:
            if model is None:
                st.error("Model not loaded. Please check server configuration.")
            else:
                try:
                    preds = model.predict(img)
                    if preds is None or len(preds) == 0:
                        st.error("Invalid prediction output.")
                    else:
                        pred_array = preds[0] if len(preds.shape) > 1 else preds
                        if np.all(np.isnan(pred_array)) or np.all(pred_array == 0):
                            st.error("Invalid prediction values.")
                        else:
                            top5 = np.argsort(pred_array)[-5:][::-1]
                            pred_label = CLASS_NAMES[int(top5[0])]
                            confidence = float(pred_array[top5[0]]) * 100
                            feature_info = get_feature_info(pred_label)
                            # --- Enhanced Result Output ---
                            st.markdown("---")
                            st.markdown("#### 🌟 Prediction Result")
                            st.success(f"**Prediction:** `{pred_label}` ({confidence:.2f}%)")
                            st.markdown(f"**🧠 Description:** {feature_info['description']}")
                            st.markdown(f"**🔖 Tags:** `{', '.join(feature_info['features'])}`")
                            st.write("**Top 5 Predictions:**")
                            for i in top5:
                                st.write(f"- {CLASS_NAMES[i]}: {float(pred_array[i])*100:.2f}%")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.text(traceback.format_exc())

st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}
.css-18e3th9 {
    background: rgba(255, 255, 255, 0.9) !important;
    backdrop-filter: blur(10px);
    border-radius: 24px;
    padding: 40px !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1), 0 0 0 1px rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    transition: all 0.3s ease;
}
.css-18e3th9:hover {
    transform: translateY(-5px);
    box-shadow: 0 30px 60px rgba(0,0,0,0.15);
}
h1, h2, h3, h4, h5 {
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}
.stButton>button {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 0.75em 2em;
    font-size: 1em;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 30px rgba(17,153,142,0.3);
}
.stFileUploader>div>div {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 12px;
    font-weight: bold;
    padding: 10px;
}
.stAlert {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 16px;
    padding: 20px;
    font-weight: bold;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)
# [END OF FILE]
