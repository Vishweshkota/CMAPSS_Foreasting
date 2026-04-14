from flask import Flask, request, jsonify, render_template

from model_loader import load_all_assets
from preprocessing import preprocess_single_row
from state import EngineStateManager
from inference import predict_rul


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


# Start the local Flask development server.
if __name__ == "__main__":
    app.run(debug=True)