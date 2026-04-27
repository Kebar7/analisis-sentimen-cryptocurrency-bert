import streamlit as st
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report

st.markdown("""
<style>

/* ===== Background Utama ===== */
[data-testid="stAppViewContainer"] {
    background: #dae4f7;
}
/* ===== Browse File Button Only (CSS Only Fix) ===== */
[data-testid="stFileUploader"] section button {
    background-color: #2563eb !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 6px 14px !important;
}

[data-testid="stFileUploader"] section button:hover {
    background-color: #1e3a8a !important;
    color: white !important;
}

/* =======================================================
   HEADER STREAMLIT FULL WHITE TEXT
   ======================================================= */

/* Semua isi header jadi putih */
[data-testid="stHeader"] * {
    color: #111827 !important;
}


[data-testid="stToolbar"] {
    background: #dae4f7;
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e40af, #2563eb);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* ===== Semua Text jadi dark soft (bukan hitam pekat) ===== */
html, body, p, span, div, label {
    color: #1f2937 !important;
}

/* ===== Judul ===== */
h1, h2, h3 {
    color: #1e3a8a !important;
    font-weight: 600;
}

/* ===== Button ===== */
.stButton>button {
    background-color: #3b82f6 !important;
    color: white !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 6px 14px !important;
}

.stButton>button:hover {
    background-color: #1e3a8a;
    transform: translateY(-1px);
}

/* ===== Download Button ===== */
.stDownloadButton>button {
    background-color: #3b82f6;
    color: white;
    border-radius: 10px;
}

/* ===== Metric Box ===== */
[data-testid="stMetric"] {
    background-color: #e0ecff;
    padding: 12px;
    border-radius: 12px;
}

/* ===== Selectbox ===== */
div[data-baseweb="select"] > div {
    background-color: #f0f6ff !important;
    border-radius: 8px;
}

/* ===== Input & Textarea ===== */
input, textarea {
    background-color: #f8fbff !important;
    border-radius: 8px !important;
}

/* ===== File uploader (hapus hitam gelap) ===== */
[data-testid="stFileUploaderDropzone"] {
    background-color: #eaf2ff !important;
    border: 2px dashed #3b82f6 !important;
    color: #1f2937 !important;
}

/* ===== Dataframe area biar nggak hitam ===== */
[data-testid="stDataFrame"] {
    background-color: white !important;
}

/* ===== Divider ===== */
hr {
    border: 1px solid #93c5fd;
}
/* ===== Dataframe Full White ===== */
[data-testid="stDataFrame"] {
    background-color: white !important;
    color: #1f2937 !important;
}

[data-testid="stDataFrame"] th {
    background-color: #dbeafe !important;
    color: #1e3a8a !important;
}

[data-testid="stDataFrame"] td {
    background-color: white !important;
    color: #111827 !important;
}

/* Scroll area dalam dataframe */
[data-testid="stDataFrame"] div {
    color: #111827 !important;
}
/* ===== Warna Text Saat User Mengetik ===== */
input,
textarea {
    color: #111827 !important;   /* hitam soft */
    -webkit-text-fill-color: #111827 !important;
}

/* =======================================================
   DATAFRAME FIX — FULL WHITE + TEXT HITAM + BORDER JELAS
   ======================================================= */

/* Container utama */
[data-testid="stDataFrame"] {
    background-color: white !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
}

/* Scroll wrapper dalam */
[data-testid="stDataFrame"] > div {
    background-color: white !important;
}

/* Header */
[data-testid="stDataFrame"] thead tr th {
    background-color: #e5edff !important;
    color: #111827 !important;
    font-weight: 600 !important;
    border: 1px solid #d1d5db !important;
}

/* Cell isi */
[data-testid="stDataFrame"] tbody tr td {
    background-color: white !important;
    color: #111827 !important;
    border: 1px solid #e5e7eb !important;
}

/* Hover row */
[data-testid="stDataFrame"] tbody tr:hover td {
    background-color: #f3f4f6 !important;
}

/* Paksa semua teks dalam grid jadi hitam */
[data-testid="stDataFrame"] * {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.sidebar-footer {
    position: fixed;
    bottom: 10px;
    left: 20px;
    font-size: 12px;
    color: white;
    opacity: 0.8;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# CLASS CONFIG
# =========================================================
class Config:
    MODELS = {
        "Skenario 1 (3 epoch, learning rate 2e-5 (best model)": {"folder_asli": "./model_skenario_baru1_siap_deploy3e5neww"},
        "Skenario 2 (5 epoch, learning rate 2e-5)": {"folder_asli": "./model_skenario_baru1_siap_deploy 5 E5"},
        "Skenario 3 (8 epoch, learning rate 2e-5)": {"folder_asli": "./model_skenario_baru1_siap_deploy8,E5"},
        "Skenario 4 (10 epoch, learning rate 2e-5)": {"folder_asli": "./model_skenario_baru10elern2e5_siap_deploy"},
        "Skenario 5 (3 epoch, learning rate 2e-6)": {"folder_asli": "./3e6"},
        "Skenario 6 (5 epoch, learning rate 2e-6)": {"folder_asli": "./model_skenario_baru1_siap_deploy,E6 5"},
        "Skenario 7 (8 epoch, learning rate 2e-6)": {"folder_asli": "./model_skenario_baru1_siap_deploy, 8 ELERN 6"},
        "Skenario 8 (10 epoch, batch 32, lr 2e-6)": {"folder_asli": "./model_skenario_baru1_siap_deploy10,6"},
        "Skenario 9 (3 epoch, batch 32, lr 2e-5, max length 60)": {"folder_asli": "./3e6-MAXLENGTH - Copy"},
        "Skenario 10 (data train cryptonews only)": {"folder_asli": "./SKENARIO 10"},
    }
    MAX_LEN = 128


# =========================================================
# CLASS UTILS
# =========================================================
class Utils:

    @staticmethod
    @st.cache_resource
    def load_model_resources(path):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = BertForSequenceClassification.from_pretrained(path)
        model.eval()
        return tokenizer, model

    @staticmethod
    def predict_batch_data(texts, tokenizer, model):

        predictions = []
        confidences = []

        progress = st.progress(0)
        status = st.empty()
        total = len(texts)

        for i, text in enumerate(texts):

            inputs = tokenizer(
                str(text),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=Config.MAX_LEN
            )

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

            pred = int(np.argmax(probs))
            predictions.append(pred)
            confidences.append(float(probs[pred]))

            progress.progress((i + 1) / total)
            status.text(f"⏳ Memproses data ke-{i+1} dari {total}")

        status.empty()
        progress.empty()

        return predictions, confidences

    @staticmethod
    def evaluate_predictions(y_true, preds):

        report_dict = classification_report(
            y_true,
            preds,
            target_names=["Negative", "Positive"],
            output_dict=True
        )

        cm = confusion_matrix(y_true, preds)

        return cm, report_dict


# ==============================================================================
# KONFIGURASI
# ==============================================================================
st.set_page_config(
    page_title="Skripsi Sentiment Analysis",
    page_icon="🎓",
    layout="wide"
)
# ======================================================================
# CUSTOM CLEAN BLUE THEME (ELEGANT VERSION)
# ======================================================================



# ==============================================================================
# MENU 1 — PREDIKSI
# ==============================================================================
# =========================================================
# CLASS STREAMLIT APP
# =========================================================
# =========================================================
# CLASS STREAMLIT APP
# =========================================================
class StreamlitApp:

    def __init__(self):
        pass  # config & css tetap di atas

    def sidebar(self):
        st.sidebar.markdown("## Sistem Analisis Sentimen")
        st.sidebar.caption("Model BERT untuk Analisis Cryptocurrency")
        menu = st.sidebar.radio(
            "Pilih Halaman:",
            [
                "1. Analisis Sentimen",
                "2. Evaluasi Model"
            ],
            key="main_menu"   # 🔥 FIX DUPLICATE ID
        )

        # FOOTER (PUNYA LU - JANGAN DIHAPUS)
        st.sidebar.markdown(
            """
            <div class="sidebar-footer">
                M. Akbar Kevin <br>
                09021282227048 <br>
                © 2026
            </div>
            """,
            unsafe_allow_html=True
        )

        return menu

    def run(self):

        menu = self.sidebar()

        # ================================
        # MENU 1 — PREDIKSI
        # ================================
        if menu == "1. Analisis Sentimen":

            st.title("Analisis Sentimen Cryptocurrency Mengunakan Model Bert")

            selected_model = st.selectbox(
                "Pilih Skenario Model:",
                list(Config.MODELS.keys()),
                key="model_prediksi"
            )

            model_path = Config.MODELS[selected_model]["folder_asli"]
            tokenizer, model = Utils.load_model_resources(model_path)

            text = st.text_area("Masukkan Kalimat:", "", height=120)

            if st.button("🔍 Analisis", type="primary"):

                if text.strip() == "":
                    st.error("Teks tidak boleh kosong")
                else:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=128
                    )

                    with torch.no_grad():
                        outputs = model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]

                    label = "POSITIF 😊" if probs[1] > probs[0] else "NEGATIF 😡"
                    conf = max(probs)

                    st.subheader(f"Hasil Prediksi: {label}")
                    st.metric("Confidence", f"{conf*100:.2f}%")
                    st.success(f"Menggunakan Model: {selected_model}")

        # ================================
        # MENU 2 — EVALUASI
        # ================================
        elif menu == "2. Evaluasi Model":

            st.title("Evaluasi Model BERT dengan data uji Cryptocurrency")
            
            selected_model = st.selectbox(
                "Pilih Skenario Model:",
                list(Config.MODELS.keys()),
                key="model_evaluasi"
            )

            model_path = Config.MODELS[selected_model]["folder_asli"]
            tokenizer, model = Utils.load_model_resources(model_path)

            uploaded_file = st.file_uploader(
                "Upload file CSV untuk Evaluasi",
                type=["csv"],
                key="upload_csv"
            )

            if uploaded_file:

                df = pd.read_csv(uploaded_file)
                st.subheader("Preview Dataset")
                st.dataframe(df.head(10), use_container_width=True)

                if "sentiment" not in df.columns:
                    st.error("Dataset harus memiliki kolom 'sentiment'")
                else:

                    jumlah_data = st.number_input(
                        "Jumlah Data Uji:",
                        min_value=1,
                        max_value=len(df),
                        value=min(50, len(df)),
                        key="jumlah_data"
                    )

                    df_test = df.iloc[:jumlah_data].copy()

                    if st.button("⚡ Jalankan Evaluasi"):

                        df_test = df.iloc[:jumlah_data].copy()

                        preds, confs = Utils.predict_batch_data(
                            df_test["text_clean"], tokenizer, model
                        )

                        df_test["Prediksi"] = ["Positif" if p == 1 else "Negatif" for p in preds]
                        df_test["Confidence"] = confs

                        st.subheader("📊 Hasil Analisis")

                        st.dataframe(
                            df_test[["text_clean", "Prediksi", "Confidence"]],
                            use_container_width=True,
                            hide_index=True
                        )

                        y_true_raw = df_test["sentiment"].astype(str).str.lower().str.strip()
                        label_map = {"negative": 0, "positive": 1}
                        y_true = y_true_raw.map(label_map)

                        cm, report_dict = Utils.evaluate_predictions(y_true, preds)

                        st.success(f"Menggunakan Model: {selected_model}")

                        st.divider()
                        st.subheader("📊 Confusion Matrix")

                        fig_cm, ax_cm = plt.subplots(figsize=(4,3))

                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            xticklabels=["Negative", "Positive"],
                            yticklabels=["Negative", "Positive"],
                            annot_kws={"size": 8},
                            ax=ax_cm
                        )

                        plt.tight_layout()
                        st.pyplot(fig_cm, use_container_width=False)

                        # ===== CLASSIFICATION REPORT =====
                        st.subheader("📊 Hasil evaluasi ")

                        report_df = pd.DataFrame(report_dict).transpose()

                        accuracy_value = report_dict.get("accuracy", None)

                        if "accuracy" in report_df.index:
                            report_df = report_df.drop("accuracy")

                        report_df = report_df.round(4)
                        report_df["accuracy"] = ""

                        if accuracy_value is not None:
                            acc_percent = accuracy_value * 100
                            report_df.loc["weighted avg", "accuracy"] = f"{acc_percent:.2f} %"

                        for row in ["macro avg", "weighted avg"]:
                            if row in report_df.index:
                                for col in ["precision", "recall", "f1-score"]:
                                    report_df.loc[row, col] = f"{report_df.loc[row, col]*100:.2f} %"

                        if "support" in report_df.columns:
                            report_df["support"] = report_df["support"].astype(int)

                        st.dataframe(report_df, use_container_width=True)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    app = StreamlitApp()
    app.run()