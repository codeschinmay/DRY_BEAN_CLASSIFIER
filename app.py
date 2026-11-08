"""Streamlit front-end: evaluate pre-trained classifiers on an uploaded test CSV.
Models are only loaded and used for prediction - nothing is retrained here."""
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from utils import compute_metrics, MODEL_NAMES, METRIC_COLS

st.set_page_config(page_title="Classifier Evaluation Lab", page_icon="📊", layout="wide")


@st.cache_resource
def load_artifacts():
    meta = json.load(open("model/meta.json"))
    le = joblib.load("model/label_encoder.joblib")
    models = {n: joblib.load(f"model/{n.lower().replace(' ', '_')}.joblib")
              for n in MODEL_NAMES}
    return meta, le, models


meta, le, models = load_artifacts()
target, features, n_classes = meta["target"], meta["features"], len(le.classes_)

st.title("📊 Classifier Evaluation Lab")
st.caption(f"Five pre-trained models · target column: **{target}** · "
           f"classes: {', '.join(meta['classes'])}")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV", type="csv")
with open("test_data.csv", "rb") as f:
    st.sidebar.download_button("Download sample test_data.csv", f, "test_data.csv", "text/csv")
choice = st.sidebar.selectbox("Select model", MODEL_NAMES)

if uploaded is None:
    st.info("👈 Upload a test CSV (all feature columns + the target column) to begin. "
            "You can download a ready-made sample from the sidebar.")
    st.stop()

# ---------------- Validate input ----------------
df = pd.read_csv(uploaded)
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required column(s): {missing}")
    st.stop()
df = df.dropna(subset=[target])
unknown = set(df[target].astype(str)) - set(meta["classes"])
if unknown:
    st.error(f"Target has unseen label(s): {sorted(unknown)}")
    st.stop()

X = df[features]
y = df[target].astype(str).map({str(c): i for i, c in enumerate(le.classes_)}).values

with st.expander(f"Preview of uploaded data ({len(df)} rows)"):
    st.dataframe(df.head(20), use_container_width=True)


def evaluate(name):
    pipe = models[name]
    pred, proba = pipe.predict(X), pipe.predict_proba(X)
    return pred, compute_metrics(y, pred, proba, n_classes)


tab1, tab2 = st.tabs([f"🔍 {choice}", "🏁 Compare all models"])

# ---------------- Selected model ----------------
with tab1:
    pred, m = evaluate(choice)
    st.subheader("Evaluation metrics")
    cols = st.columns(len(METRIC_COLS))
    for c, k in zip(cols, METRIC_COLS):
        c.metric(k, f"{m[k]:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, pred, labels=range(n_classes))
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        rep = classification_report(y, pred, labels=range(n_classes),
                                    target_names=[str(c) for c in le.classes_],
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

# ---------------- Comparison ----------------
with tab2:
    rows = [{"ML Model Name": n, **evaluate(n)[1]} for n in MODEL_NAMES]
    comp = pd.DataFrame(rows).set_index("ML Model Name")
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c6f6d5"),
                 use_container_width=True)
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
    st.success(f"Best MCC on this file: **{comp['MCC'].idxmax()}**")
