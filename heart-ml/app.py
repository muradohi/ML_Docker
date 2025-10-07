import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from joblib import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Heart Disease ML", page_icon="❤️", layout="wide")
import streamlit as st

st.title("❤️ Heart Disease – End-to-End (EDA → Train → Predict)")

DATA_DIR = Path("data")

# ---------- helpers ----------
def find_csv(datadir: Path) -> Path:
    preferred = [
        "heart-disease-cleveland-uci.csv",
        "heart.csv",
        "cleveland.csv",
        "heart_cleveland_upload.csv",
    ]
    for name in preferred:
        p = datadir / name
        if p.exists():
            return p
    csvs = sorted(datadir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {datadir.resolve()}")
    return csvs[0]

def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df = df.replace("?", np.nan)
    # find target
    for t in ["target", "num", "class", "condition"]:
        if t in df.columns:
            y = pd.to_numeric(df[t], errors="coerce")
            df["target_bin"] = (y > 0).astype(int)  # Cleveland: 0 vs 1..4
            break
    else:
        raise ValueError("Could not find target column (expected: target/num/class/condition)")
    return df

def build_pipeline(model_name: str, num_cols, cat_cols) -> Pipeline:
    numeric_tf = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_tf = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")),
               ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    pre = ColumnTransformer(
        transformers=[("num", numeric_tf, num_cols), ("cat", categorical_tf, cat_cols)]
    )
    if model_name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000)
    else:
        clf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])

# ---------- sidebar: data + training options ----------
st.sidebar.header("Data & Training")
csv_choice = st.sidebar.text_input("CSV path (optional)", value="")
test_size = st.sidebar.slider("Test size", 0.15, 0.4, 0.25, 0.01)
model_name = st.sidebar.selectbox("Model", ["Random Forest", "Logistic Regression"])
seed = st.sidebar.number_input("Random seed", 0, 10_000, 42)

# load data
try:
    csv_path = Path(csv_choice) if csv_choice.strip() else find_csv(DATA_DIR)
    df_raw = pd.read_csv(csv_path)
    df = standardize_dataframe(df_raw)
    st.success(f"Loaded: {csv_path.name}  ({len(df)} rows)")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# choose features present
numeric_candidates = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
categorical_candidates = ["sex", "fbs", "exang", "cp", "restecg", "slope", "thal"]
num_cols = [c for c in numeric_candidates if c in df.columns]
cat_cols = [c for c in categorical_candidates if c in df.columns]
if not (num_cols or cat_cols):
    st.error("No known feature columns found in the CSV.")
    st.stop()

# ---------- EDA ----------
with st.expander("👀 Quick EDA", expanded=True):
    c1, c2 = st.columns([1, 1])
    with c1:
        st.write("Sample")
        st.dataframe(df.head(10), use_container_width=True)
        class_counts = df["target_bin"].value_counts().rename({0: "No disease", 1: "Disease"})
        st.write("Class balance")
        st.bar_chart(class_counts)
    with c2:
        st.write("Numeric distributions")
        sel = st.selectbox("Select numeric feature", options=num_cols, index=0)
        fig = px.histogram(df, x=sel, color=df["target_bin"].map({0:"No disease",1:"Disease"}), barmode="overlay", nbins=40, opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

# ---------- Train ----------
X = df[num_cols + cat_cols].copy()
y = df["target_bin"].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed, stratify=y
)
pipe = build_pipeline(model_name, num_cols, cat_cols)

train_btn = st.button("🚀 Train model", type="primary")

if train_btn:
    with st.spinner("Training..."):
        pipe.fit(X_train, y_train)
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3, output_dict=False)
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])

    st.subheader("Results")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("5-fold CV F1 (train)", f"{cv.mean():.3f}", f"±{cv.std():.3f}")
    with m2:
        st.metric("F1 (test)", f"{f1:.3f}")

    cm_df = pd.DataFrame(cm, index=["True: No", "True: Yes"], columns=["Pred: No", "Pred: Yes"])
    fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix", aspect="auto")
    st.plotly_chart(fig_cm, use_container_width=True)
    with st.expander("Classification report"):
        st.code(report)

    # Save model + schema in memory and offer download
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True, parents=True)
    model_name_tag = "rf" if model_name == "Random Forest" else "logreg"
    model_path = models_dir / f"heart_{model_name_tag}.joblib"
    dump(pipe, model_path)

    schema = {"numeric": num_cols, "categorical": cat_cols}
    schema_path = models_dir / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f)

    st.success(f"Saved: {model_path} and {schema_path}")
    with open(model_path, "rb") as f:
        st.download_button("⬇️ Download model (.joblib)", f, file_name=model_path.name)
    with open(schema_path, "rb") as f:
        st.download_button("⬇️ Download feature schema (.json)", f, file_name=schema_path.name)

    st.divider()

    # ---------- Single-patient prediction UI ----------
    st.subheader("🩺 Predict for a single patient")
    with st.form("predict_form"):
        # Build widgets dynamically using present columns
        inputs = {}
        # numeric
        ncols = st.columns(3)
        for i, col in enumerate(num_cols):
            with ncols[i % 3]:
                default = float(np.nanmedian(pd.to_numeric(df[col], errors="coerce"))) if df[col].notna().any() else 0.0
                inputs[col] = st.number_input(col, value=float(default))
        # categorical/binary
        cat_widgets = st.columns(3)
        # define reasonable choices if present
        choices = {
            "sex": [0, 1],           # 1=male, 0=female (usual encoding)
            "fbs": [0, 1],           # fasting blood sugar > 120 mg/dl
            "exang": [0, 1],         # exercise induced angina
            "cp": [0,1,2,3],         # chest pain type
            "restecg": [0,1,2],      # resting ECG
            "slope": [0,1,2],        # ST segment slope
            "thal": [0,1,2,3,6,7],   # varies by source; keep flexible
        }
        for i, col in enumerate(cat_cols):
            with cat_widgets[i % 3]:
                opts = choices.get(col, sorted(df[col].dropna().unique().tolist()))
                inputs[col] = st.selectbox(col, options=opts, index=0)

        submitted = st.form_submit_button("Predict")
    if train_btn and submitted:
        # Build one-row dataframe in feature order
        row = {**{c: inputs[c] for c in num_cols}, **{c: inputs[c] for c in cat_cols}}
