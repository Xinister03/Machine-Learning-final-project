import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. KONFIGURASI HALAMAN & DESIGN SYSTEM (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Ames Housing — Enterprise AI Dashboard",
    page_icon="🏠",
    layout="wide",
)

# Custom Styling CSS (Basis Desain Gradien Ungu-Biru Premium)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }

    .section-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 2.5rem 0 1.5rem;
    }

    /* Card Output Harga */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
    }
    
    .result-card .label {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    
    .result-card .price {
        color: #ffffff;
        font-size: 3.4rem;
        font-weight: 700;
        margin: 0;
    }

    /* Kotak Insight Otomatis */
    .insight-box {
        background-color: rgba(102, 126, 234, 0.06);
        border-left: 4px solid #764ba2;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
    }

    /* Blok Kontainer Pengembang di Tab About Us */
    .about-main-container {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1rem;
    }
    .about-main-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.75rem;
    }
    .dev-main-card {
        background-color: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1.2rem;
        line-height: 1.7;
    }

    .system-footer {
        text-align: center;
        color: #555;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 2. ADVANCED MULTI-MODEL PIPELINE ENGINE
# ==========================================
@st.cache_resource
def PIPELINE_PRODUCTION_ENGINE():
    try:
        df = pd.read_csv("train.csv")
    except FileNotFoundError:
        df = pd.read_csv("dataset/train.csv")
        
    feature_names = ["Gr Liv Area", "Overall Qual", "Garage Cars", "Total Bsmt SF", "Year Built"]
    target_name = "SalePrice"
    
    df_selected = df[feature_names + [target_name]].copy()
    
    Q1 = df_selected[target_name].quantile(0.25)
    Q3 = df_selected[target_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df_selected[
        (df_selected[target_name] >= lower_bound) & (df_selected[target_name] <= upper_bound)
    ].copy()
    
    X = df_clean[feature_names]
    y = df_clean[target_name]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    pipeline_xgb = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    pipeline_xgb.fit(X_train, y_train)
    
    pipeline_rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    pipeline_rf.fit(X_train, y_train)
    
    pipeline_ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    pipeline_ridge.fit(X_train, y_train)
    
    pipeline_knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5, weights="distance"))
    ])
    pipeline_knn.fit(X_train, y_train)
    
    y_pred_xgb = pipeline_xgb.predict(X_test)
    y_pred_rf = pipeline_rf.predict(X_test)
    y_pred_ridge = pipeline_ridge.predict(X_test)
    y_pred_knn = pipeline_knn.predict(X_test)
    
    metrics_data = {
        "XGBoost Regressor": {
            "r2": r2_score(y_test, y_pred_xgb), "mae": mean_absolute_error(y_test, y_pred_xgb), "rmse": np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        },
        "Random Forest": {
            "r2": r2_score(y_test, y_pred_rf), "mae": mean_absolute_error(y_test, y_pred_rf), "rmse": np.sqrt(mean_squared_error(y_test, y_pred_rf))
        },
        "Ridge Regression": {
            "r2": r2_score(y_test, y_pred_ridge), "mae": mean_absolute_error(y_test, y_pred_ridge), "rmse": np.sqrt(mean_squared_error(y_test, y_pred_ridge))
        },
        "K-Nearest Neighbors (KNN)": {
            "r2": r2_score(y_test, y_pred_knn), "mae": mean_absolute_error(y_test, y_pred_knn), "rmse": np.sqrt(mean_squared_error(y_test, y_pred_knn))
        }
    }
    
    importances_data = {
        "XGBoost Regressor": dict(zip(feature_names, pipeline_xgb.named_steps["model"].feature_importances_)),
        "Random Forest": dict(zip(feature_names, pipeline_rf.named_steps["model"].feature_importances_)),
        "Ridge Regression": dict(zip(feature_names, pipeline_ridge.named_steps["model"].coef_)),
        "K-Nearest Neighbors (KNN)": {f: 0.0 for f in feature_names}
    }
    
    models_map = {
        "XGBoost Regressor": pipeline_xgb,
        "Random Forest": pipeline_rf,
        "Ridge Regression": pipeline_ridge,
        "K-Nearest Neighbors (KNN)": pipeline_knn
    }
    
    return models_map, df_clean, metrics_data, importances_data, feature_names

models_dict, df_clean, metrics_dict, importances_dict, feature_list = PIPELINE_PRODUCTION_ENGINE()


# ==========================================
# 3. INTERFACE NAVIGATION (SIDEBAR)
# ==========================================
with st.sidebar:
    st.markdown("### Main Menu Navigation")
    # Urutan Menu Baru yang sudah disesuaikan secara linier
    active_dashboard_tab = st.radio(
        "Pilih Menu Dashboard:",
        options=["About Us", "EDA & Dataset", "Training & Evaluation"],
        label_visibility="collapsed"
    )


# ==========================================
# 4. MAIN PROGRAM LAYOUT
# ==========================================

# ------------------------------------------
# HALAMAN 1: ABOUT US
# ------------------------------------------
if active_dashboard_tab == "About Us":
    st.subheader("Informasi Proyek dan Anggota Tim")
    
    st.markdown(
        """
        <div class="about-main-container">
            <div class="about-main-header">Proyek Profil</div>
            <p style='color: #ccc; line-height: 1.6; font-size: 0.95rem;'>
                Project ini bertujuan memprediksi harga rumah berdasarkan fitur seperti luas, kualitas, dan fasilitas menggunakan model machine learning.
                Dengan mengandalkan modul standardisasi Pipeline terintegrasi, arsitektur dashboard mampu menyajikan komparasi evaluasi 
                metrik performa empat rumpun algoritma mutakhir secara objektif, aman, dan real-time.
                <br><br>
                <b>Dataset Utama:</b> Ames Housing Dataset<br>
                <b>Teknologi:</b> Python, Streamlit, Scikit-learn, XGBoost, Plotly Express
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.write("")
    st.markdown("#### Tim Pengembang Dokumen")
    
    col_dev1, col_dev2 = st.columns(2, gap="large")
    with col_dev1:
        st.markdown(
            """
            <div class="dev-main-card">
                <span style='color: #667eea; font-weight: bold; font-size: 1.1rem;'>Anggota 1</span><br>
                <b>Nama Lengkap:</b> Satya Herlambang Kurniawan<br>
                <b>NIM / Kode Registrasi:</b> 2802428002<br>
                <b>Afiliasi Kampus:</b> BINUS University Student<br>
                <b>Kontak Email:</b> <a href='mailto:satya.kurniawan@binus.ac.id' style='color: #667eea;'>satya.kurniawan@binus.ac.id</a>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_dev2:
        st.markdown(
            """
            <div class="dev-main-card">
                <span style='color: #764ba2; font-weight: bold; font-size: 1.1rem;'>Anggota 2</span><br>
                <b>Nama Lengkap:</b> Ivan Novanto Bastian<br>
                <b>NIM / Kode Registrasi:</b> 2802428002<br>
                <b>Afiliasi Kampus:</b> BINUS University Student<br>
                <b>Kontak Email:</b> <a href='mailto:ivan.bastian@binus.ac.id' style='color: #764ba2;'>ivan.bastian@binus.ac.id</a>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------
# HALAMAN 2: EDA & DATASET
# ------------------------------------------
elif active_dashboard_tab == "EDA & Dataset":
    st.subheader("Exploratory Data Analysis Dashboard")
    
    sub_histogram, sub_scatterplot, sub_boxplot, sub_model_info = st.tabs([
        "1. Distribution", 
        "2. Relationships", 
        "3. Categories", 
        "4. Model Info"
    ])
    
    with sub_histogram:
        st.markdown("#### Sebaran Frekuensi Variabel Target (SalePrice)")
        fig_hist = px.histogram(
            df_clean, x="SalePrice", nbins=40,
            title="Histogram Distribusi Harga Jual Rumah di Ames",
            color_discrete_sequence=["#667eea"], template="plotly_white"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        
        q1_p = df_clean["SalePrice"].quantile(0.25)
        q3_p = df_clean["SalePrice"].quantile(0.75)
        
        st.markdown("##### Peta Segmentasi Kelas Harga Pasar")
        col_l, col_m, col_h = st.columns(3)
        col_l.metric("Segmen Ekonomi (Low-End)", f"< ${q1_p:,.0f}")
        col_m.metric("Segmen Menengah (Mid-Market)", f"${q1_p:,.0f} - ${q3_p:,.0f}")
        col_h.metric("Segmen Premium (High-End)", f"> ${q3_p:,.0f}")

    with sub_scatterplot:
        st.markdown("#### Pola Korelasi Linier Fitur Numerik vs Target")
        target_feature = st.selectbox(
            "Pilih Variabel Finansial/Dimensi untuk Ditinjau Hubungannya:",
            options=["Gr Liv Area", "Total Bsmt SF", "Year Built"]
        )
        fig_scatter = px.scatter(
            df_clean, x=target_feature, y="SalePrice", trendline="ols",
            title=f"Scatter Plot Analisis Pola Hubungan: {target_feature} vs SalePrice",
            color_discrete_sequence=["#764ba2"], opacity=0.5, template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        current_r_val = df_clean[target_feature].corr(df_clean["SalePrice"])
        st.success(f"Insight Korelasi: Indikator koefisien korelasi Pearson menunjukkan angka r = {current_r_val:.2f}. Ini mengonfirmasi korelasi linear positif yang signifikan.")

    with sub_boxplot:
        st.markdown("#### Rentang Variansi Harga Properti antar Kategori Kualitas")
        fig_box = px.box(
            df_clean, x="Overall Qual", y="SalePrice",
            title="Box Plot Pemetaan Distribusi Rentang Harga Properti Berdasarkan Skor Kualitas",
            color="Overall Qual", color_discrete_sequence=px.colors.sequential.Purples_r,
            template="plotly_white"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with sub_model_info:
        st.markdown("#### Lembar Perbandingan Performa Algoritma")
        eda_active_model = st.selectbox(
            "Pilih Model untuk Meninjau Feature Importance lokal:",
            options=["XGBoost Regressor", "Random Forest", "Ridge Regression", "K-Nearest Neighbors (KNN)"]
        )
        
        col_chart_cell, col_table_cell = st.columns([1.1, 1])
        with col_chart_cell:
            current_corr_matrix = df_clean[feature_list + ["SalePrice"]].corr()
            fig_heatmap = px.imshow(
                current_corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Correlation Heatmap Matrix"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
        with col_table_cell:
            comparison_records = []
            for m_name, m_metrics in metrics_dict.items():
                comparison_records.append({
                    "Algoritma Model": m_name, "R² Score": f"{m_metrics['r2']*100:.2f}%", "MAE": f"${m_metrics['mae']:,.2f}", "RMSE": f"${m_metrics['rmse']:,.2f}"
                })
            st.dataframe(pd.DataFrame(comparison_records), hide_index=True, use_container_width=True)
            
            st.write("")
            if eda_active_model == "K-Nearest Neighbors (KNN)":
                st.info("KNN memprediksi harga berdasarkan kedekatan jarak matriks, tidak memiliki bobot kepengaruhan fitur linier bawaan.")
            else:
                current_importances = importances_dict[eda_active_model]
                df_importance_plot = pd.DataFrame({"Nama Indikator Properti": list(current_importances.keys()), "Skor Bobot Pengaruh": list(current_importances.values())}).sort_values(by="Skor Bobot Pengaruh", ascending=True)
                chart_title_label = "Standardized Coefficients" if "Ridge" in eda_active_model else "Feature Importance"
                fig_importance_bar = px.bar(df_importance_plot, x="Skor Bobot Pengaruh", y="Nama Indikator Properti", orientation="h", title=f"Karakteristik Pengaruh Variabel ({chart_title_label})", color="Skor Bobot Pengaruh", color_continuous_scale="Plasma", template="plotly_white")
                st.plotly_chart(fig_importance_bar, use_container_width=True)

# ------------------------------------------
# HALAMAN 3: TRAINING & EVALUATION (PREDICTION SYSTEM)
# ------------------------------------------
elif active_dashboard_tab == "Training & Evaluation":
    st.markdown("### Konfigurasi Model Operasional")
    active_model = st.selectbox(
        "Pilih Algoritma Operasional untuk Prediksi:",
        options=["XGBoost Regressor", "Random Forest", "Ridge Regression", "K-Nearest Neighbors (KNN)"]
    )
    st.markdown("---")

    col_input, col_display = st.columns([1, 1.2], gap="large")
    
    with col_input:
        st.subheader("Parameter Properti")
        gr_liv_area = st.number_input("Luas Ruang Tamu Atas Tanah (Gr Liv Area - sqft)", min_value=100, max_value=7000, value=1500, step=50)
        overall_qual = st.slider("Skor Kualitas Material Bangunan (Overall Qual 1-10)", min_value=1, max_value=10, value=6)
        garage_cars = st.number_input("Kapasitas Tampung Garasi (Garage Cars - Mobil)", min_value=0, max_value=5, value=2, step=1)
        total_bsmt_sf = st.number_input("Luas Total Ruang Bawah Tanah (Total Bsmt SF - sqft)", min_value=0, max_value=6000, value=1000, step=50)
        year_built = st.number_input("Tahun Konstruksi Rumah (Year Built)", min_value=1800, max_value=2026, value=2000, step=1)
        
        st.write("")
        predict_btn = st.button("Hitung Estimasi Nilai Jual", use_container_width=True, type="primary")

    with col_display:
        st.subheader(f"Kalkulasi Nilai Properti ({active_model})")
        st.markdown(f"<p style='color: #888; font-style: italic; margin-top: -10px; margin-bottom: 20px;'>Aplikasi saat ini berjalan menggunakan model: {active_model}.</p>", unsafe_allow_html=True)
            
        input_is_valid = True
        if total_bsmt_sf > gr_liv_area * 1.5:
            st.warning("Anomali Validasi Logika: Proporsi luas basement terdeteksi tidak wajar (melebihi 150% dari luas bangunan utama).")
        if year_built > 2026:
            st.error("Validasi Mutlak Gagal: Tahun pembangunan tidak valid karena melampaui ambang batas tahun berjalan (2026).")
            input_is_valid = False
            
        if predict_btn and input_is_valid:
            user_input_df = pd.DataFrame(
                [[gr_liv_area, overall_qual, garage_cars, total_bsmt_sf, year_built]],
                columns=feature_list
            )
            
            current_pipeline = models_dict[active_model]
            predicted_value = current_pipeline.predict(user_input_df)[0]
            
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Estimasi Hasil Harga Prediksi</div>
                    <p class="price">${predicted_value:,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            current_mae = metrics_dict[active_model]["mae"]
            floor_price = max(0, predicted_value - current_mae)
            ceil_price = predicted_value + current_mae
            
            st.write("")
            st.markdown("##### Batas Toleransi Rentang Harga Realistis (Confidence Range)")
            st.info(f"Berdasarkan nilai Mean Absolute Error (MAE) dari algoritma {active_model}, harga sesungguhnya diproyeksikan berada pada kisaran: ${floor_price:,.0f} — ${ceil_price:,.0f}.")
            
            st.markdown("##### Analisis Nilai Tambah Properti (Smart Insights)")
            insights_list = []
            if overall_qual >= 8:
                insights_list.append("Premium Material Quality: Penilaian kualitas material superior bertindak sebagai instrumen utama pendorong nilai apresiasi harga tinggi.")
            elif overall_qual <= 4:
                insights_list.append("Koreksi Kualitas Struktural: Komponen material bangunan di bawah rata-rata berisiko kuat memicu depresiasi harga pasar.")
            if gr_liv_area > df_clean["Gr Liv Area"].quantile(0.75):
                insights_list.append("Dimensi Spasial Premium: Luas ruang tamu berada di atas kuartil 75% pasar Ames, memberikan daya tawar premium.")
            if year_built >= 2000:
                insights_list.append("Modern Utility Infrastructure: Bangunan milenium baru (tahun 2000 ke atas) menjamin ketersediaan sistem utilitas yang prima.")
            
            if not insights_list:
                insights_list.append("Profil spesifikasi properti berada dalam batas standar rata-rata pasar perumahan.")
                
            for ins in insights_list:
                st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)
                
            st.caption("Prediksi ini merupakan hasil pendekatan matematis berbasis tren data historis dan tidak bersifat mengikat secara absolut.")
        elif not predict_btn and input_is_valid:
            st.info("Silakan konfigurasikan parameter spesifikasi rumah di panel kiri, kemudian eksekusi dengan klik tombol 'Predict House Price'.")

# Global Copyright Footer Paling Bawah
st.markdown('<div class="system-footer">© 2026 Ames Housing Enterprise AI Dashboard | Designed for Academic Examination Presentation</div>', unsafe_allow_html=True)