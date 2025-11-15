# streamlit_app.py
import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
from datetime import datetime

MODEL_PATH = Path("models/rf_model_with_features.joblib")  # update if needed

def load_model_and_features(path):
    p = Path(path)
    if not p.exists():
        return None, None, f"Model file not found: {p}"

    try:
        loaded = joblib.load(p)
    except Exception as e:
        return None, None, f"Failed to load model file {p}: {e}"

    if isinstance(loaded, dict):
        model = None
        features = None
        for k in ("model", "pipeline", "estimator"):
            if k in loaded:
                model = loaded[k]
                break
        if model is None:
            for v in loaded.values():
                if hasattr(v, "predict"):
                    model = v
                    break
        for fk in ("features", "feature_names", "feature_list"):
            if fk in loaded:
                features = loaded[fk]
                break
        return model, features, None
    else:
        if hasattr(loaded, "predict"):
            return loaded, None, None
        else:
            return None, None, f"Loaded object from {p} has no .predict()"

@st.cache_resource
def cached_model_loader():
    return load_model_and_features(MODEL_PATH)

def build_single_input_row(feature_names):
    """
    Build a dictionary of user inputs for each expected feature.
    This uses heuristics to create sensible widgets:
      - if 'hour' or 'day' in name -> int inputs
      - if 'temp' or 'temperature' -> float inputs
      - otherwise -> float input
    Customize as needed for your real features.
    """
    vals = {}
    for f in feature_names:
        fname = f.lower()
        if "hour" in fname or "day" in fname or "weekday" in fname:
            vals[f] = st.number_input(f, min_value=0, max_value=23, value=12, step=1)
        elif "temp" in fname or "temperature" in fname:
            vals[f] = st.number_input(f, value=20.0, format="%.2f")
        elif "occup" in fname or "people" in fname or "occupancy" in fname:
            vals[f] = st.number_input(f, min_value=0, value=1, step=1)
        elif "timestamp" in fname or "date" in fname:
            txt = st.text_input(f"{f} (YYYY-MM-DD HH:MM:SS)", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            vals[f] = txt
        else:
            # default numeric
            vals[f] = st.number_input(f, value=0.0, format="%.4f")
    return vals

def validate_and_prepare(df, expected_features):
    """
    Ensure df contains expected features (order doesn't matter).
    If the pipeline expects certain dtypes or timestamp parsing, do it here
    or keep that logic within the pipeline so the app stays simple.
    """
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {missing}"
    # reorder columns to expected order
    df = df[expected_features].copy()
    return df, None

def main():
    st.title("Energy Consumption Predictor (Streamlit)")
    st.markdown("Load model and predict. This app expects a saved model dict with keys `model` and `features` (feature list).")

    model, features, err = cached_model_loader()

    if err:
        st.error(err)
        st.info("Fix: ensure you've saved the joblib file and path is correct (e.g., models/rf_model_with_features.joblib).")
        uploaded_model = st.file_uploader("Or upload a model file (joblib/pkl)", type=["joblib","pkl","sav","bin"])
        if uploaded_model is not None:
            try:
                loaded = joblib.load(uploaded_model)
            except Exception as e:
                st.error(f"Upload failed to load: {e}")
                st.stop()
            if isinstance(loaded, dict) and "model" in loaded:
                model = loaded["model"]
                features = loaded.get("features", None)
                st.success("Model (dict) uploaded. Extracted 'model' key.")
            elif hasattr(loaded, "predict"):
                model = loaded
                features = None
                st.success("Model uploaded and loaded.")
            else:
                st.error("Uploaded file did not contain a model with .predict().")
                st.stop()
        else:
            st.stop()

    if model is None:
        st.error("Model could not be loaded.")
        st.stop()

    st.success("Model loaded successfully.")
    if features:
        st.info(f"Model expects these features: {features}")

    st.header("Input options")
    mode = st.radio("Choose input mode", ("Single example", "Upload CSV"))

    if mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV of feature rows", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            if features:
                df_prepared, err = validate_and_prepare(df, features)
                if err:
                    st.error(err)
                    st.stop()
            else:
                df_prepared = df
            st.write("Preview:", df_prepared.head())
            if st.button("Predict for uploaded CSV"):
                try:
                    preds = model.predict(df_prepared)
                    df_out = df_prepared.copy()
                    df_out["prediction"] = preds
                    st.write(df_out)
                    st.download_button("Download predictions", df_out.to_csv(index=False), "predictions.csv")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    else:  # Single example
        if features:
            inputs = build_single_input_row(features)
            if st.button("Predict single example"):
                # Build df in expected order
                df_row = pd.DataFrame([inputs])
                # If any timestamp-like strings are present and your pipeline expects parsed datetimes,
                # either parse here or rely on pipeline to parse.
                try:
                    if features:
                        df_row, err = validate_and_prepare(df_row, features)
                        if err:
                            st.error(err); st.stop()
                    preds = model.predict(df_row)
                    st.metric("Predicted energy consumption", f"{preds[0]:.4f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("Model loaded but no feature list baked into model metadata. Upload a CSV or edit the app to provide feature names.")
            txt = st.text_input("Enter comma-separated feature names (order matters):")
            if txt:
                features = [s.strip() for s in txt.split(",") if s.strip()]
                inputs = build_single_input_row(features)
                if st.button("Predict (using provided features)"):
                    df_row = pd.DataFrame([inputs])
                    try:
                        preds = model.predict(df_row)
                        st.metric("Predicted energy consumption", f"{preds[0]:.4f}")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
