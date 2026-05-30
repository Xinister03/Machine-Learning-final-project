import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Ames Housing — Prediksi Harga Rumah",
    page_icon="🏠",
    layout="centered",
)

# ──────────────────────────────────────────────
# Load Model
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Memuat model Linear Regression dari file pkl."""
    return joblib.load("housing_model.pkl")

model = load_model()

# ──────────────────────────────────────────────
# Custom CSS — tampilan lebih bersih & modern
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Google Font ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Header area ---------- */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.05rem;
    }

    /* ---------- Divider ---------- */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin: 1.5rem 0;
    }

    /* ---------- Result card ---------- */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.30);
    }
    .result-card .label {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .result-card .price {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.82rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🏠 Ames Housing Price Predictor</h1>
        <p>Masukkan spesifikasi rumah, lalu klik <b>Predict</b> untuk melihat estimasi harga.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Input Form
# ──────────────────────────────────────────────
st.subheader("📋 Spesifikasi Rumah")

col1, col2 = st.columns(2)

with col1:
    gr_liv_area = st.number_input(
        "Luas Ruang Tamu (sqft)",
        min_value=200,
        max_value=6000,
        value=1500,
        step=50,
        help="GrLivArea — luas bangunan di atas tanah (above grade living area).",
    )

    overall_qual = st.number_input(
        "Kualitas Keseluruhan (1-10)",
        min_value=1,
        max_value=10,
        value=6,
        step=1,
        help="OverallQual — penilaian kualitas material & finishing rumah secara keseluruhan.",
    )

    garage_cars = st.number_input(
        "Kapasitas Garasi (mobil)",
        min_value=0,
        max_value=5,
        value=2,
        step=1,
        help="GarageCars — jumlah mobil yang bisa ditampung garasi.",
    )

with col2:
    total_bsmt_sf = st.number_input(
        "Luas Basement (sqft)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50,
        help="TotalBsmtSF — total luas area basement.",
    )

    year_built = st.number_input(
        "Tahun Dibangun",
        min_value=1872,
        max_value=2026,
        value=2000,
        step=1,
        help="YearBuilt — tahun pembangunan rumah.",
    )

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
predict_btn = st.button("🔮 Predict House Price", use_container_width=True, type="primary")

if predict_btn:
    # Susun DataFrame input sesuai urutan fitur saat training
    input_df = pd.DataFrame(
        [[gr_liv_area, overall_qual, garage_cars, total_bsmt_sf, year_built]],
        columns=["Gr Liv Area", "Overall Qual", "Garage Cars", "Total Bsmt SF", "Year Built"],
    )

    # Prediksi
    prediction = model.predict(input_df)[0]

    # Tampilkan hasil
    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">Estimasi Harga Rumah</div>
            <p class="price">${prediction:,.0f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detail input dalam expander
    with st.expander("📊 Detail Input yang Digunakan"):
        st.dataframe(
            input_df.rename(columns={
                "Gr Liv Area": "Luas Ruang Tamu (sqft)",
                "Overall Qual": "Kualitas (1-10)",
                "Garage Cars": "Garasi (mobil)",
                "Total Bsmt SF": "Luas Basement (sqft)",
                "Year Built": "Tahun Dibangun",
            }),
            hide_index=True,
            use_container_width=True,
        )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
        Model: Linear Regression · Dataset: Ames Housing · Powered by Streamlit & Scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
