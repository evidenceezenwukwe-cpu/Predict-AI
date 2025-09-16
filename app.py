# app.py
import os
import io
import pickle
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr

MODEL_PATH = "eurusd_predictor.h5"
SCALER_PATH = "eurusd_scaler.pkl"

# --- Load model ---
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise RuntimeError(f"Could not load model '{MODEL_PATH}': {e}")

# --- Load scaler (joblib or pickle) ---
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

# --- Get model input shape (helpful for users) ---
def get_input_shape_str(m):
    s = m.input_shape
    if isinstance(s, list):
        s = s[0]
    return tuple([None if x is None else int(x) for x in s])

expected_shape = get_input_shape_str(model)

# capture model.summary() as string
def model_summary_str(m):
    buf = io.StringIO()
    m.summary(print_fn=lambda x: buf.write(x + "\n"))
    return buf.getvalue()

model_info = f"Model input shape: {expected_shape}\n\nModel summary:\n{model_summary_str(model)}"

# --- helpers to prepare input data ---
def parse_text_numbers(text):
    # parse comma-separated numbers
    parts = [p.strip() for p in text.strip().split(",") if p.strip() != ""]
    return np.array([float(p) for p in parts], dtype=float)

def prepare_array(arr):
    """
    arr: 1D numpy array OR 2D numpy array (rows x features)
    Returns X ready for model.predict
    """
    arr = np.asarray(arr, dtype=float)
    # if 2D (rows x features), keep as is (we will predict for each row)
    if arr.ndim == 2:
        X = arr
    else:
        # 1D
        # expected_shape is like (None, features) or (None, timesteps, features)
        if len(expected_shape) == 2:
            features = expected_shape[1]
            if arr.size != features:
                raise ValueError(f"Model expects {features} features (1D input). You passed {arr.size}.")
            X = arr.reshape(1, features)
        elif len(expected_shape) == 3:
            timesteps = expected_shape[1]
            features = expected_shape[2]
            if arr.size == timesteps * features:
                X = arr.reshape(1, timesteps, features)
            elif arr.size == features:
                # assume single timestep
                X = arr.reshape(1, 1, features)
            else:
                raise ValueError(f"Model expects shape (timesteps,features)={ (timesteps,features) } but input length {arr.size} doesn't match.")
        else:
            # fallback: reshape to (1, n)
            X = arr.reshape(1, -1)

    # apply scaler if present and if data is 2D (rows x features)
    if scaler is not None:
        try:
            # scaler.transform needs 2D array; only apply when X is 2D (not 3D sequences)
            if X.ndim == 2:
                X = scaler.transform(X)
        except Exception:
            # if scaler fails, skip scaling but warn
            pass
    return X

# --- Prediction function for Gradio ---
def predict(file_obj, text_input):
    """
    file_obj: uploaded CSV file (or None)
    text_input: comma-separated numbers input (or empty)
    """
    # 1) get data from file if provided
    try:
        if file_obj is not None:
            # pandas can read file-like objects
            df = pd.read_csv(file_obj.name) if isinstance(file_obj, gradio.components.File) else pd.read_csv(file_obj)
            arr = df.values
            # if multiple rows, take the last row (most recent)
            if arr.ndim == 2:
                row = arr[-1]
            else:
                row = arr
            X = prepare_array(row)
        elif text_input and text_input.strip() != "":
            nums = parse_text_numbers(text_input)
            X = prepare_array(nums)
        else:
            return {"error": "Please upload a CSV or paste comma-separated numbers."}
    except Exception as e:
        return {"error": f"Input error: {e}"}

    # predict
    try:
        pred = model.predict(X)
        # flatten result to list
        out = pred.ravel().tolist()
        return {"prediction": out, "model_info": model_info}
    except Exception as e:
        return {"error": f"Prediction error: {e}", "model_info": model_info}

# --- Build Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# EURUSD Predictor (demo)\nUpload a CSV (one row of features) or paste comma-separated numbers.\nThe app will show model input shape and a prediction.")
    with gr.Row():
        csv_input = gr.File(label="Upload CSV (one row = most recent)", file_types=["csv"], interactive=True)
        text_input = gr.Textbox(label="Or paste comma-separated numbers (e.g. 1.234, 1.235, ...)", placeholder="1.234, 1.235, 1.233, ...", lines=2)
    predict_btn = gr.Button("Run prediction")
    output = gr.JSON(label="Result")
    model_info_box = gr.Textbox(label="Model info (readonly)", value=model_info, interactive=False)
    predict_btn.click(fn=predict, inputs=[csv_input, text_input], outputs=output)

if __name__ == "__main__":
    demo.launch()