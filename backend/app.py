from flask import Flask, request, jsonify, render_template

from model_loader import load_all_assets
from preprocessing import preprocess_single_row
from state import EngineStateManager
from inference import predict_rul

import numpy as np
import pandas as pd

# Build a fixed 30-row model window from an uploaded CSV file.
def build_window_from_uploaded_file(df, tag, engine_id, rows_to_use, artifacts):
    setting_cols = artifacts["setting_cols"]
    sensor_cols = artifacts["sensor_cols"]
    window_size = artifacts["window_size"]

    required_cols = ["time_cycles"] + setting_cols + sensor_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded file is missing required columns: {missing_cols}")

    # If the file contains multiple engines, filter by selected engine_id.
    if "engine_id" in df.columns:
        df = df[df["engine_id"] == engine_id].copy()
        if df.empty:
            raise ValueError(f"No rows found for engine_id={engine_id} in uploaded file")

    df = df.sort_values("time_cycles").reset_index(drop=True)

    rows_to_use = int(rows_to_use)
    if rows_to_use <= 0:
        raise ValueError("rows_to_use must be greater than 0")

    selected_df = df.tail(min(rows_to_use, len(df))).copy()
    if selected_df.empty:
        raise ValueError("No usable rows found in uploaded file")

    processed_rows = []

    for _, row in selected_df.iterrows():
        raw_row = {
            "tag": tag,
            "engine_id": int(row["engine_id"]) if "engine_id" in selected_df.columns else int(engine_id),
            "time_cycles": int(row["time_cycles"]),
            "setting_1": float(row["setting_1"]),
            "setting_2": float(row["setting_2"]),
            "setting_3": float(row["setting_3"]),
            "sensor_1": float(row["sensor_1"]),
            "sensor_2": float(row["sensor_2"]),
            "sensor_3": float(row["sensor_3"]),
            "sensor_4": float(row["sensor_4"]),
            "sensor_5": float(row["sensor_5"]),
            "sensor_6": float(row["sensor_6"]),
            "sensor_7": float(row["sensor_7"]),
            "sensor_8": float(row["sensor_8"]),
            "sensor_9": float(row["sensor_9"]),
            "sensor_10": float(row["sensor_10"]),
            "sensor_11": float(row["sensor_11"]),
            "sensor_12": float(row["sensor_12"]),
            "sensor_13": float(row["sensor_13"]),
            "sensor_14": float(row["sensor_14"]),
            "sensor_15": float(row["sensor_15"]),
            "sensor_16": float(row["sensor_16"]),
            "sensor_17": float(row["sensor_17"]),
            "sensor_18": float(row["sensor_18"]),
            "sensor_19": float(row["sensor_19"]),
            "sensor_20": float(row["sensor_20"]),
            "sensor_21": float(row["sensor_21"])
        }

        preprocess_result = preprocess_single_row(raw_row, artifacts)
        processed_rows.append(
            preprocess_result["processed_row"].iloc[0].to_numpy(dtype=np.float32)
        )

    rows_selected_before_windowing = len(processed_rows)
    padded = False

    if len(processed_rows) < window_size:
        pad_count = window_size - len(processed_rows)
        first_row = processed_rows[0]
        processed_rows = [first_row.copy() for _ in range(pad_count)] + processed_rows
        padded = True
    else:
        processed_rows = processed_rows[-window_size:]

    window = np.stack(processed_rows, axis=0)

    return window, {
        "rows_available_after_filter": int(len(df)),
        "rows_selected_before_windowing": int(rows_selected_before_windowing),
        "final_window_rows": int(window_size),
        "padded": padded,
        "last_cycle_used": int(selected_df["time_cycles"].iloc[-1])
    }


# Create the Flask app and load all models/artifacts once at startup.
app = Flask(__name__)

assets = load_all_assets()
state_manager = EngineStateManager(
    window_size=assets["preprocessing_artifacts"]["window_size"]
)


# Simple homepage for browser testing.
@app.route("/")
def home():
    return render_template("index.html")


# Health check route to confirm the backend is running.
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "message": "Flask backend is running",
        "available_models": ["lr", "rf", "lstm"],
        "window_size": assets["preprocessing_artifacts"]["window_size"]
    })


# Return basic model and input information for the client.
@app.route("/models", methods=["GET"])
def models():
    return jsonify({
        "available_models": ["lr", "rf", "lstm"],
        "required_raw_fields": (
            ["tag", "engine_id", "time_cycles"]
            + assets["preprocessing_artifacts"]["setting_cols"]
            + assets["preprocessing_artifacts"]["sensor_cols"]
        ),
        "feature_cols": assets["preprocessing_artifacts"]["feature_cols"],
        "window_size": assets["preprocessing_artifacts"]["window_size"]
    })


# Predict from a full processed 30-cycle window sent directly by the client.
@app.route("/predict/window", methods=["POST"])
def predict_window():
    try:
        payload = request.get_json()

        if payload is None:
            return jsonify({"ok": False, "error": "Request body must be valid JSON"}), 400

        model_name = payload.get("model")
        window = payload.get("window")
        tag = payload.get("tag")
        engine_id = payload.get("engine_id")

        if model_name not in ["lr", "rf", "lstm"]:
            return jsonify({"ok": False, "error": "Unsupported model name"}), 400

        if window is None:
            return jsonify({"ok": False, "error": "Missing window data"}), 400

        window_size = assets["preprocessing_artifacts"]["window_size"]
        feature_count = len(assets["preprocessing_artifacts"]["feature_cols"])

        if len(window) != window_size:
            return jsonify({
                "ok": False,
                "error": f"Window must contain exactly {window_size} rows"
            }), 400

        for row in window:
            if len(row) != feature_count:
                return jsonify({
                    "ok": False,
                    "error": f"Each row must contain exactly {feature_count} values"
                }), 400

        import numpy as np
        window_array = np.array(window, dtype=np.float32)

        predicted_rul = predict_rul(model_name, window_array, assets)

        return jsonify({
            "ok": True,
            "mode": "window",
            "model": model_name,
            "tag": tag,
            "engine_id": engine_id,
            "window_ready": True,
            "predicted_rul": predicted_rul
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# Predict from one raw sensor row at a time using rolling state per engine.
@app.route("/predict/stream", methods=["POST"])
def predict_stream():
    try:
        raw_row = request.get_json()

        if raw_row is None:
            return jsonify({"ok": False, "error": "Request body must be valid JSON"}), 400

        model_name = raw_row.get("model")
        if model_name not in ["lr", "rf", "lstm"]:
            return jsonify({"ok": False, "error": "Unsupported model name"}), 400

        preprocess_result = preprocess_single_row(
            raw_row,
            assets["preprocessing_artifacts"]
        )

        processed_row = preprocess_result["processed_row"]

        tag = raw_row["tag"]
        engine_id = raw_row["engine_id"]

        current_length = state_manager.add_processed_row(
            tag=tag,
            engine_id=engine_id,
            processed_row=processed_row
        )

        if not state_manager.is_window_ready(tag, engine_id):
            remaining = assets["preprocessing_artifacts"]["window_size"] - current_length

            return jsonify({
                "ok": True,
                "mode": "stream",
                "model": model_name,
                "tag": tag,
                "engine_id": engine_id,
                "window_ready": False,
                "cycles_collected": current_length,
                "message": f"Need {remaining} more cycles before prediction"
            })

        window = state_manager.get_window(tag, engine_id)
        predicted_rul = predict_rul(model_name, window, assets)

        return jsonify({
            "ok": True,
            "mode": "stream",
            "model": model_name,
            "tag": tag,
            "engine_id": engine_id,
            "window_ready": True,
            "cycles_collected": current_length,
            "predicted_rul": predicted_rul
        })

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Predict with all three models at once using one shared streaming update.
@app.route("/predict/stream/all", methods=["POST"])
def predict_stream_all():
    try:
        raw_row = request.get_json()

        if raw_row is None:
            return jsonify({"ok": False, "error": "Request body must be valid JSON"}), 400

        preprocess_result = preprocess_single_row(
            raw_row,
            assets["preprocessing_artifacts"]
        )

        processed_row = preprocess_result["processed_row"]

        tag = raw_row["tag"]
        engine_id = raw_row["engine_id"]

        current_length = state_manager.add_processed_row(
            tag=tag,
            engine_id=engine_id,
            processed_row=processed_row
        )

        if not state_manager.is_window_ready(tag, engine_id):
            remaining = assets["preprocessing_artifacts"]["window_size"] - current_length

            return jsonify({
                "ok": True,
                "mode": "stream_all",
                "tag": tag,
                "engine_id": engine_id,
                "window_ready": False,
                "cycles_collected": current_length,
                "message": f"Need {remaining} more cycles before prediction"
            })

        window = state_manager.get_window(tag, engine_id)

        predictions = {
            "lr": predict_rul("lr", window, assets),
            "rf": predict_rul("rf", window, assets),
            "lstm": predict_rul("lstm", window, assets)
        }

        return jsonify({
            "ok": True,
            "mode": "stream_all",
            "tag": tag,
            "engine_id": engine_id,
            "window_ready": True,
            "cycles_collected": current_length,
            "predictions": predictions
        })

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Reset one engine's rolling history.
@app.route("/engines/reset", methods=["POST"])
def reset_engine():
    try:
        payload = request.get_json()

        if payload is None:
            return jsonify({"ok": False, "error": "Request body must be valid JSON"}), 400

        tag = payload.get("tag")
        engine_id = payload.get("engine_id")

        if tag is None or engine_id is None:
            return jsonify({"ok": False, "error": "Both tag and engine_id are required"}), 400

        state_manager.reset_engine(tag, engine_id)

        return jsonify({
            "ok": True,
            "message": f"Reset state for engine {engine_id} in {tag}"
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# Reset all in-memory engine histories.
@app.route("/engines/reset_all", methods=["POST"])
def reset_all():
    try:
        state_manager.reset_all()

        return jsonify({
            "ok": True,
            "message": "Reset all engine states"
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Predict RUL from an uploaded CSV file of sensor readings.
@app.route("/predict/file", methods=["POST"])
def predict_file():
    try:
        uploaded_file = request.files.get("file")
        model_name = request.form.get("model")
        tag = request.form.get("tag")
        engine_id = request.form.get("engine_id")
        rows_to_use = request.form.get("rows_to_use")

        if uploaded_file is None or uploaded_file.filename == "":
            return jsonify({"ok": False, "error": "A CSV file is required"}), 400

        if model_name not in ["lr", "rf", "lstm"]:
            return jsonify({"ok": False, "error": "Unsupported model name"}), 400

        if tag not in ["FD001", "FD002", "FD003", "FD004"]:
            return jsonify({"ok": False, "error": "Unsupported tag"}), 400

        if engine_id is None or rows_to_use is None:
            return jsonify({"ok": False, "error": "engine_id and rows_to_use are required"}), 400

        engine_id = int(engine_id)
        rows_to_use = int(rows_to_use)

        df = pd.read_csv(uploaded_file)

        window, info = build_window_from_uploaded_file(
            df=df,
            tag=tag,
            engine_id=engine_id,
            rows_to_use=rows_to_use,
            artifacts=assets["preprocessing_artifacts"]
        )

        predicted_rul = predict_rul(model_name, window, assets)

        return jsonify({
            "ok": True,
            "mode": "file",
            "model": model_name,
            "tag": tag,
            "engine_id": engine_id,
            "rows_requested": rows_to_use,
            "rows_available_after_filter": info["rows_available_after_filter"],
            "rows_selected_before_windowing": info["rows_selected_before_windowing"],
            "final_window_rows": info["final_window_rows"],
            "padded": info["padded"],
            "last_cycle_used": info["last_cycle_used"],
            "predicted_rul": predicted_rul
        })

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# Start the local Flask development server.
if __name__ == "__main__":
    app.run(debug=True)