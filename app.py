import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Demand & Stockout Predictor", layout="wide")


@st.cache_resource
def load_artifacts():
    a = "artifacts/"
    return {
        "clf": joblib.load(a + "stockout_model.pkl"),
        "scaler": joblib.load(a + "scaler.pkl"),
        "enc": joblib.load(a + "encoders.pkl"),
        "clf_cols": joblib.load(a + "clf_features.pkl"),
        "clf_def": joblib.load(a + "clf_defaults.pkl"),
        "reg": joblib.load(a + "demand_model.pkl"),
        "reg_cols": joblib.load(a + "reg_features.pkl"),
        "reg_def": joblib.load(a + "reg_defaults.pkl"),
    }


def collect_inputs(cols, defaults, enc, key):
    """Build one input widget per feature and return a 1-row DataFrame."""
    values = {}
    grid = st.columns(3)
    for i, c in enumerate(cols):
        with grid[i % 3]:
            if c in enc:  # categorical column: show the original text labels
                choice = st.selectbox(c, list(enc[c].classes_), key=f"{key}_{c}")
                values[c] = int(enc[c].transform([choice])[0])
            else:
                values[c] = st.number_input(
                    c, value=float(defaults[c]), key=f"{key}_{c}"
                )
    return pd.DataFrame([values], columns=cols)


art = load_artifacts()

st.title("Retail Demand Forecasting & Stockout Risk")
st.caption("Random Forest models trained on 2024-2025 daily product data")

tab1, tab2 = st.tabs(["Stockout risk (next 7 days)", "Demand forecast (units)"])

with tab1:
    st.subheader("Will this product hit a stockout in the next 7 days?")
    row = collect_inputs(art["clf_cols"], art["clf_def"], art["enc"], "clf")
    if st.button("Predict stockout risk"):
        X = art["scaler"].transform(row)  # same scaling used in training
        prob = art["clf"].predict_proba(X)[0][1]
        pred = art["clf"].predict(X)[0]
        st.metric("Stockout probability", f"{prob:.1%}")
        st.progress(float(prob))
        if pred == 1:
            st.error("High risk: consider reordering soon.")
        else:
            st.success("Low risk: stock level looks fine.")

with tab2:
    st.subheader("How many units will customers demand?")
    row = collect_inputs(art["reg_cols"], art["reg_def"], art["enc"], "reg")
    if st.button("Forecast demand"):
        units = art["reg"].predict(row)[0]  # tree model: no scaling needed
        st.metric("Predicted units demanded", f"{units:.1f}")
