from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import torch

app = Flask(__name__)
CORS(app)

# load all four models
model = joblib.load("xgboost_model.pkl")

# def extract_features(rbp_np, scc_np):
#     """Same feature extraction as model_training.py — must be identical"""
#     rbp_mean = np.mean(rbp_np, axis=1)
#     scc_mean = np.mean(scc_np, axis=1)
#     rbp_std  = np.std(rbp_np, axis=1, ddof=1)
#     scc_std  = np.std(scc_np, axis=1, ddof=1)

#     rbp_mean_flat = rbp_mean.reshape(len(rbp_mean), -1)
#     scc_mean_flat = scc_mean.reshape(len(scc_mean), -1)
#     rbp_std_flat  = rbp_std.reshape(len(rbp_std), -1)
#     scc_std_flat  = scc_std.reshape(len(scc_std), -1)

#     X = np.hstack([rbp_mean_flat, scc_mean_flat, rbp_std_flat, scc_std_flat])
#     return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# def majority_vote(y_pred, y_proba, subs):
#     """Combine window-level predictions into one per subject"""
#     results = []
#     for s in np.unique(subs):
#         mask = subs == s
#         vote       = int(np.round(np.mean(y_pred[mask])))
#         confidence = float(np.mean(y_proba[mask]))
#         results.append({
#             "subject":    s,
#             "prediction": "AD" if vote == 1 else "CN",
#             "confidence": round(confidence * 100, 2)  # as a percentage
#         })
#     return results

# @app.route("/predict", methods=["POST"])
# def predict():
#     # Check that a file was actually uploaded
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]

#     if not file.filename.endswith(".pt"):
#         return jsonify({"error": "File must be a .pt dataset file"}), 400

#     try:
#         # Load the uploaded .pt file
#         data = torch.load(file, weights_only=False)

#         X_rbp  = data["X_rbp"].numpy()
#         X_scc  = data["X_scc"].numpy()
#         groups = data["groups"]

#         # Extract features
#         X = extract_features(X_rbp, X_scc)

#         # Run model
#         pred  = model.predict(X)
#         proba = model.predict_proba(X)[:, 1]

#         # Aggregate to subject level
#         results = majority_vote(pred, proba, groups)

#         return jsonify({"results": results})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)