# ============================================================
# app.py – Aplikasi Streamlit
# Klasifikasi Ekspresi Wajah: Senang, Marah, Sedih
# CNN from Scratch vs Transfer Learning (EfficientNetB0)
# ============================================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt
import time

# ---- Konfigurasi Halaman ----
st.set_page_config(
    page_title="Klasifikasi Ekspresi Wajah",
    page_icon="😊",
    layout="wide"
)

# ---- Konstanta ----
CLASS_NAMES = ["senang", "marah", "sedih"]
EMOJI_MAP   = {"senang": "😊", "marah": "😠", "sedih": "😢"}
IMG_SIZE    = 128
MODEL_CNN_PATH = "model_cnn_scratch.keras"
MODEL_TL_PATH  = "model_efficientnetb0_tl.keras"

# ---- Load Model (cached) ----
@st.cache_resource
def load_models():
    """Muat kedua model satu kali dan simpan di cache."""
    model_cnn = keras.models.load_model(MODEL_CNN_PATH)
    model_tl  = keras.models.load_model(MODEL_TL_PATH)
    return model_cnn, model_tl

# ---- Preprocessing ----
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize dan normalisasi gambar untuk input model."""
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_array   = np.array(img_resized) / 255.0
    return np.expand_dims(img_array, axis=0)

# ---- Prediksi ----
def predict(model, img_array: np.ndarray):
    """Jalankan prediksi dan kembalikan kelas serta probabilitas."""
    preds     = model.predict(img_array, verbose=0)[0]
    pred_idx  = np.argmax(preds)
    return CLASS_NAMES[pred_idx], float(preds[pred_idx]) * 100, preds

# ============================================================
# TAMPILAN UTAMA APLIKASI
# ============================================================
st.title("😊 Klasifikasi Ekspresi Wajah")
st.markdown("""
> **Proyek Akhir – Artificial Intelligence for Education**
> Perbandingan CNN from Scratch vs Transfer Learning (EfficientNetB0)
""")
st.divider()

# ---- Sidebar ----
with st.sidebar:
    st.header("ℹ️ Informasi Model")
    st.markdown("""
    **Kelas yang dikenali:**
    - 😊 Senang (Happy)
    - 😠 Marah (Angry)
    - 😢 Sedih (Sad)

    **Model:**
    - CNN from Scratch
    - Transfer Learning (EfficientNetB0)

    **Input:** Gambar wajah (JPG/PNG)
    """)
    st.divider()
    st.caption("Dibuat untuk Proyek Akhir AI for Education")

# ---- Load Model ----
with st.spinner("⏳ Memuat model... mohon tunggu"):
    try:
        model_cnn, model_tl = load_models()
        st.success("✅ Kedua model berhasil dimuat!")
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        st.stop()

# ---- Upload Gambar ----
st.subheader("📤 Upload Gambar Wajah")
uploaded_file = st.file_uploader(
    "Pilih gambar wajah (JPG atau PNG):",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Tampilkan gambar dan hasil prediksi
    img = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess_image(img)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Prediksi CNN
    with col2:
        st.subheader("🧠 CNN from Scratch")
        with st.spinner("Memproses..."):
            t0 = time.time()
            pred_cnn, conf_cnn, probs_cnn = predict(model_cnn, img_array)
            t_cnn = (time.time() - t0) * 1000

        emoji = EMOJI_MAP.get(pred_cnn, "🤔")
        st.metric(label="Prediksi", value=f"{emoji} {pred_cnn.capitalize()}")
        st.metric(label="Confidence", value=f"{conf_cnn:.2f}%")
        st.caption(f"⏱️ Waktu inferensi: {t_cnn:.1f} ms")

        fig_cnn, ax_cnn = plt.subplots(figsize=(5, 3))
        colors_cnn = ["#2ecc71" if c == pred_cnn else "#bdc3c7" for c in CLASS_NAMES]
        ax_cnn.barh([c.capitalize() for c in CLASS_NAMES],
                    probs_cnn * 100, color=colors_cnn)
        ax_cnn.set_xlabel("Probabilitas (%)")
        ax_cnn.set_xlim(0, 105)
        for i, v in enumerate(probs_cnn * 100):
            ax_cnn.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
        ax_cnn.set_title("Distribusi Probabilitas CNN", fontsize=10)
        st.pyplot(fig_cnn, clear_figure=True)

    # Prediksi Transfer Learning
    with col3:
        st.subheader("⚡ Transfer Learning (EfficientNetB0)")
        with st.spinner("Memproses..."):
            t0 = time.time()
            pred_tl, conf_tl, probs_tl = predict(model_tl, img_array)
            t_tl = (time.time() - t0) * 1000

        emoji_tl = EMOJI_MAP.get(pred_tl, "🤔")
        st.metric(label="Prediksi", value=f"{emoji_tl} {pred_tl.capitalize()}")
        st.metric(label="Confidence", value=f"{conf_tl:.2f}%")
        st.caption(f"⏱️ Waktu inferensi: {t_tl:.1f} ms")

        fig_tl, ax_tl = plt.subplots(figsize=(5, 3))
        colors_tl = ["#e74c3c" if c == pred_tl else "#bdc3c7" for c in CLASS_NAMES]
        ax_tl.barh([c.capitalize() for c in CLASS_NAMES],
                   probs_tl * 100, color=colors_tl)
        ax_tl.set_xlabel("Probabilitas (%)")
        ax_tl.set_xlim(0, 105)
        for i, v in enumerate(probs_tl * 100):
            ax_tl.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
        ax_tl.set_title("Distribusi Probabilitas Transfer Learning", fontsize=10)
        st.pyplot(fig_tl, clear_figure=True)

    # Perbandingan hasil
    st.divider()
    st.subheader("📊 Perbandingan Prediksi")
    col_a, col_b, col_c = st.columns(3)
    col_a.info(f"**CNN:** {EMOJI_MAP.get(pred_cnn,'')} {pred_cnn.capitalize()} ({conf_cnn:.1f}%)")
    col_b.info(f"**Transfer Learning:** {EMOJI_MAP.get(pred_tl,'')} {pred_tl.capitalize()} ({conf_tl:.1f}%)")
    if pred_cnn == pred_tl:
        col_c.success("✅ Kedua model sepakat!")
    else:
        col_c.warning("⚠️ Kedua model berbeda prediksi.")

else:
    st.info("👆 Upload gambar wajah untuk memulai prediksi.")
    st.markdown("""
    **Tips untuk hasil terbaik:**
    - Pastikan wajah terlihat jelas dan tidak tertutup
    - Gunakan pencahayaan yang cukup
    - Ambil foto dari jarak dekat (close-up)
    """)
