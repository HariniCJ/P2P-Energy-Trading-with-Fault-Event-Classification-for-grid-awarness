# app.py
import os
import json
import numpy as np
import torch
from flask import Flask, request, render_template, jsonify
import traceback # Import traceback module for detailed error logging

# Import the model definition
from model_def import ConvNet # Assuming model_def.py is in the same directory

# --- Configuration ---
ARTIFACTS_DIR = 'flask_artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'best_model.pth')
NORM_STATS_PATH = os.path.join(ARTIFACTS_DIR, 'normalization_stats.npz')
LABELS_PATH = os.path.join(ARTIFACTS_DIR, 'labels.json')
SAMPLES_INFO_PATH = os.path.join(ARTIFACTS_DIR, 'samples_info.json')
SAMPLE_DATA_DIR = os.path.join(ARTIFACTS_DIR, 'sample_data')
FAULT_DESCRIPTIONS_PATH = os.path.join(ARTIFACTS_DIR, 'fault_descriptions.json')

# --- Initialization ---
app = Flask(__name__)
print("Flask app initialized.")

# --- Load Model and Artifacts (with detailed startup logging) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"STARTUP: Using device: {device}")

id_to_label = {}
try:
    print(f"STARTUP: Attempting to load label mapping from {LABELS_PATH}...")
    with open(LABELS_PATH, 'r') as f:
        id_to_label_str = json.load(f)
        id_to_label = {int(k): v for k, v in id_to_label_str.items()}
    print(f"STARTUP: Label mapping loaded successfully. Found {len(id_to_label)} labels.")
except FileNotFoundError:
    print(f"STARTUP ERROR: Label mapping file not found at {LABELS_PATH}")
except Exception as e:
    print(f"STARTUP ERROR: Loading label mapping from {LABELS_PATH}: {e}")

fault_descriptions = {}
try:
    print(f"STARTUP: Attempting to load fault descriptions from {FAULT_DESCRIPTIONS_PATH}...")
    with open(FAULT_DESCRIPTIONS_PATH, 'r') as f:
        fault_descriptions = json.load(f)
    print(f"STARTUP: Fault descriptions loaded successfully.")
except FileNotFoundError:
    print(f"STARTUP WARNING: Fault descriptions file not found at {FAULT_DESCRIPTIONS_PATH}.")
except Exception as e:
    print(f"STARTUP ERROR: Loading fault descriptions from {FAULT_DESCRIPTIONS_PATH}: {e}")

num_classes = len(id_to_label)
if num_classes == 0:
    print("STARTUP WARNING: num_classes is 0. Model cannot be initialized properly.")

model = None
if num_classes > 0:
    try:
        print(f"STARTUP: Initializing ConvNet model structure with {num_classes} classes...")
        model = ConvNet(nc=num_classes)
        print(f"STARTUP: Model structure initialized.")
    except Exception as e:
        print(f"STARTUP ERROR: Initializing model structure: {e}")
else:
     print("STARTUP: Skipping model structure initialization as num_classes is 0.")

if model:
    try:
        print(f"STARTUP: Attempting to load model state_dict from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"STARTUP: Model state loaded and set to eval mode.")
    except FileNotFoundError:
         print(f"STARTUP ERROR: Model file not found at {MODEL_PATH}. Model is unusable.")
         model = None
    except Exception as e:
        print(f"STARTUP ERROR: Loading model state dict from {MODEL_PATH}: {e}")
        model = None

mean, std = 0, 1
try:
    print(f"STARTUP: Attempting to load normalization stats from {NORM_STATS_PATH}...")
    norm_data = np.load(NORM_STATS_PATH)
    mean = norm_data['mean']
    std = norm_data['std']
    if mean.ndim == 1: mean = mean.reshape(1, -1, 1)
    if std.ndim == 1: std = std.reshape(1, -1, 1)
    if mean.shape != (1,3,1) and mean.size == 3: mean = mean.reshape(1,3,1)
    if std.shape != (1,3,1) and std.size == 3: std = std.reshape(1,3,1)
    print(f"STARTUP: Normalization stats loaded. Mean shape: {mean.shape}, Std shape: {std.shape}")
except FileNotFoundError:
    print(f"STARTUP ERROR: Normalization stats file not found at {NORM_STATS_PATH}. Using defaults.")
except Exception as e:
    print(f"STARTUP ERROR: Loading normalization stats from {NORM_STATS_PATH}: {e}. Using defaults.")

samples_info = {}
sorted_sample_files = []
try:
    print(f"STARTUP: Attempting to load sample info from {SAMPLES_INFO_PATH}...")
    with open(SAMPLES_INFO_PATH, 'r') as f:
        samples_info = json.load(f)
    print(f"STARTUP: Sample info loaded successfully. Found {len(samples_info)} samples.")
    sorted_sample_files = sorted(samples_info.keys())
except FileNotFoundError:
     print(f"STARTUP WARNING: Sample info file not found at {SAMPLES_INFO_PATH}.")
except Exception as e:
    print(f"STARTUP ERROR: Loading sample info from {SAMPLES_INFO_PATH}: {e}")

print("--- STARTUP CHECKS COMPLETE ---")
if not id_to_label: print("CRITICAL STARTUP ISSUE: id_to_label is empty.")
if not model: print("CRITICAL STARTUP ISSUE: model is None (not loaded).")


# --- Helper Functions ---
def preprocess_input(waveform_np):
    # print("PREPROCESS: Entered preprocess_input") # Optional: very verbose
    if waveform_np.shape != (3, 726):
        if waveform_np.size == 3 * 726:
             try: waveform_np = waveform_np.reshape(3, 726)
             except Exception as e: raise ValueError(f"Input waveform has size {waveform_np.size} but cannot be reshaped to (3, 726)") from e
        else: raise ValueError(f"Input waveform has incorrect shape {waveform_np.shape} and size {waveform_np.size}, expected (3, 726)")

    local_mean = mean
    local_std = std
    if local_mean.shape != (1, 3, 1):
        try: local_mean = local_mean.reshape(1, 3, 1)
        except: raise ValueError(f"Internal Error: Mean could not be reshaped to (1,3,1), current shape {local_mean.shape}")
    if local_std.shape != (1, 3, 1):
        try: local_std = local_std.reshape(1, 3, 1)
        except: raise ValueError(f"Internal Error: Std could not be reshaped to (1,3,1), current shape {local_std.shape}")

    normalized_waveform = (waveform_np - local_mean) / (local_std + 1e-7)

    if normalized_waveform.shape != (3, 726):
         try: normalized_waveform = normalized_waveform.reshape(3, 726)
         except Exception as e: raise ValueError(f"Could not reshape normalized waveform to (3, 726) after broadcasting. Shape was {normalized_waveform.shape}") from e

    try: waveform_tensor = torch.tensor(normalized_waveform, dtype=torch.float32)
    except Exception as e: raise ValueError("Failed to convert normalized numpy array to torch tensor") from e

    if waveform_tensor.ndim == 2 and waveform_tensor.shape == (3, 726):
         final_tensor = waveform_tensor.unsqueeze(0)
    elif waveform_tensor.ndim == 3 and waveform_tensor.shape == (1, 3, 726):
         final_tensor = waveform_tensor
    else: raise ValueError(f"Tensor has unexpected shape {waveform_tensor.shape} before adding batch dimension. Expected 2D (3, 726).")

    if final_tensor.shape != (1, 3, 726):
         raise ValueError(f"Preprocessing failed: Final tensor shape is {final_tensor.shape}, but expected (1, 3, 726).")
    # print("PREPROCESS: Exiting preprocess_input successfully") # Optional
    return final_tensor

def get_p2p_event_impact(predicted_label_str, sender_peer_id=None):
    # print(f"GET_IMPACT: Processing label: {predicted_label_str}, sender: {sender_peer_id}") # Optional
    impact_info = {
        "event_type": "Unknown Event",
        "highlight_keys": ["microgrid-hub"],
        "is_fault_condition": False,
        "p2p_disruption_message": "P2P transactions may be affected by this grid event."
    }
    if not predicted_label_str or predicted_label_str.startswith("Unknown"): return impact_info
    label_lower = predicted_label_str.lower()
    is_fault = "internal" in label_lower or "external_fault" in label_lower
    impact_info["is_fault_condition"] = is_fault

    if "internal" in label_lower:
        impact_info["event_type"] = "Internal Grid Equipment Fault"
        if "power_transformer" in label_lower:
            impact_info["highlight_keys"] = ["microgrid-hub"]
            impact_info["p2p_disruption_message"] = "Severe fault in main grid interface/transformer. P2P transfers likely disrupted or islanded."
        elif "ispar" in label_lower:
            impact_info["highlight_keys"] = ["microgrid-hub"]
            impact_info["p2p_disruption_message"] = "Fault in local microgrid regulation/transformation equipment. P2P transfers likely disrupted."
        else:
             impact_info["highlight_keys"] = ["microgrid-hub"]
             impact_info["p2p_disruption_message"] = "Internal grid fault detected. P2P transfers may be unstable or disrupted."
    elif "transient" in label_lower:
        impact_info["event_type"] = "Grid Transient Disturbance"
        if "capacitor_switching" in label_lower or "ferroresonance" in label_lower:
            impact_info["highlight_keys"] = ["microgrid-hub", "lines-p2p"]
            impact_info["p2p_disruption_message"] = "Grid transient detected. May cause temporary instability for P2P transfers."
        elif "external_fault" in label_lower:
            impact_info["highlight_keys"] = ["microgrid-hub", "lines-p2p"]
            impact_info["p2p_disruption_message"] = "External fault with CT saturation detected. Supply to local P2P microgrid may be compromised."
        elif "magnetising_inrush" in label_lower or "sympathetic_inrush" in label_lower:
            impact_info["highlight_keys"] = ["microgrid-hub"]
            impact_info["event_type"] = "Transformer Inrush (Non-Fault)"
            impact_info["p2p_disruption_message"] = "Transformer inrush current. Usually temporary. P2P stability should recover."
        elif "non-linear_load_switching" in label_lower:
            if sender_peer_id and sender_peer_id in ["prosumer-a", "prosumer-b", "micro-industry", "ev-hub", "consumer-home", "local-business"]:
                impact_info["highlight_keys"] = [sender_peer_id]
            else:
                impact_info["highlight_keys"] = ["consumer-load", "ev-hub"]
            impact_info["p2p_disruption_message"] = "Non-linear load switching. May cause localized power quality issues for P2P."
        else:
            impact_info["highlight_keys"] = ["lines-p2p", "microgrid-hub"]
            impact_info["p2p_disruption_message"] = "General grid transient. P2P may experience temporary disturbances."
    else:
        impact_info["event_type"] = "Unclassified Grid Event"
        impact_info["p2p_disruption_message"] = "Unclassified grid event. Impact on P2P uncertain."

    if is_fault and sender_peer_id:
        if sender_peer_id not in impact_info["highlight_keys"]:
            impact_info["highlight_keys"] = [sender_peer_id] + impact_info["highlight_keys"]
        impact_info["p2p_disruption_message"] = f"Grid fault! Transfer from {sender_peer_id.replace('-', ' ').title()} likely disrupted. {impact_info['p2p_disruption_message']}"
    # print(f"GET_IMPACT: Returning: {impact_info}") # Optional
    return impact_info

# --- Routes ---
@app.route('/')
def index():
    print("ROUTE /: Request received for index page.")
    sample_options = []
    if samples_info and sorted_sample_files:
         try:
             sample_options = [
                 {"filename": fname, "display_name": samples_info[fname]['true_label_str']}
                 for fname in sorted_sample_files if fname in samples_info
             ]
         except Exception as e:
              print(f"ROUTE /: Error creating sample_options: {e}")
    # print(f"ROUTE /: Sending {len(sample_options)} sample options to template.") # Optional
    return render_template('index.html', sample_options=sample_options)

@app.route('/predict', methods=['POST'])
def predict():
    print("\n--- PREDICT ROUTE: Request Received ---")
    try:
        print("PREDICT: Step 1 - Checking model availability.")
        if not model:
             print("PREDICT ERROR: Model is not loaded or unusable.")
             return jsonify({"error": "Model not loaded. Cannot perform prediction."}), 500
        print("PREDICT: Model is available.")

        print("PREDICT: Step 2 - Checking request type.")
        if not request.is_json:
            print("PREDICT ERROR: Request is not JSON.")
            return jsonify({"error": "Request must be JSON"}), 400
        print("PREDICT: Request is JSON.")

        print("PREDICT: Step 3 - Getting JSON data from request.")
        data = request.get_json()
        print(f"PREDICT: Received data: {data}")

        selected_sample_filename = data.get('sample_filename')
        sender_peer = data.get('sender_peer')
        receiver_peer = data.get('receiver_peer')
        print(f"PREDICT: Parsed - Sample: {selected_sample_filename}, Sender: {sender_peer}, Receiver: {receiver_peer}")

        print("PREDICT: Step 4 - Validating inputs.")
        if not selected_sample_filename:
            print("PREDICT ERROR: No sample filename provided.")
            return jsonify({"error": "No sample filename provided in request"}), 400
        if not sender_peer or not receiver_peer:
            print("PREDICT ERROR: Sender or Receiver peer not provided.")
            return jsonify({"error": "Sender and Receiver peers must be selected"}), 400
        print("PREDICT: Inputs validated.")

        print("PREDICT: Step 5 - Constructing sample file path.")
        sample_path = os.path.join(SAMPLE_DATA_DIR, selected_sample_filename)
        print(f"PREDICT: Sample path: {sample_path}")

        print("PREDICT: Step 6 - Checking if sample file exists.")
        if not os.path.exists(sample_path):
            print(f"PREDICT ERROR: Sample file not found at {sample_path}")
            return jsonify({"error": f"Sample file not found: {selected_sample_filename}"}), 404
        print("PREDICT: Sample file exists.")

        print("PREDICT: Step 7 - Loading waveform numpy array.")
        waveform_np = np.load(sample_path)
        print(f"PREDICT: Waveform loaded. Shape: {waveform_np.shape}")

        print("PREDICT: Step 8 - Preprocessing input waveform.")
        input_tensor = preprocess_input(waveform_np.copy())
        print(f"PREDICT: Waveform preprocessed. Tensor shape: {input_tensor.shape}")

        print("PREDICT: Step 9 - Moving tensor to device.")
        input_tensor = input_tensor.to(device)
        print(f"PREDICT: Tensor moved to device: {device}")

        print("PREDICT: Step 10 - Performing model inference.")
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_prob, predicted_idx = torch.max(probabilities, 1)
        print("PREDICT: Model inference complete.")

        predicted_label_id = predicted_idx.item()
        predicted_label_str = id_to_label.get(predicted_label_id, f"Unknown Label ID: {predicted_label_id}")
        confidence = predicted_prob.item()
        print(f"PREDICT: Predicted Label ID: {predicted_label_id}, String: {predicted_label_str}, Confidence: {confidence:.2%}")

        print("PREDICT: Step 11 - Getting true label and fault description.")
        true_label_str = samples_info.get(selected_sample_filename, {}).get('true_label_str', 'N/A')
        default_desc = fault_descriptions.get("DEFAULT", {"description": "N/A", "causes": ["N/A"]})
        fault_info_content = fault_descriptions.get(predicted_label_str, default_desc)
        description = fault_info_content.get("description", default_desc["description"])
        causes = fault_info_content.get("causes", default_desc["causes"])
        print(f"PREDICT: True Label: {true_label_str}. Description and causes obtained.")

        print("PREDICT: Step 12 - Getting P2P event impact details.")
        p2p_impact_details = get_p2p_event_impact(predicted_label_str, sender_peer)
        print(f"PREDICT: P2P impact details: {p2p_impact_details}")

        print("PREDICT: Step 13 - Preparing JSON response.")
        response_data = {
            "selected_sample": selected_sample_filename,
            "sender_peer": sender_peer,
            "receiver_peer": receiver_peer,
            "predicted_label": predicted_label_str,
            "confidence": f"{confidence:.2%}",
            "true_label": true_label_str,
            "description": description,
            "causes": causes,
            "p2p_impact_details": p2p_impact_details,
            "waveform_data": waveform_np.tolist()
        }
        print("PREDICT: Response data prepared. Sending JSON response.")
        return jsonify(response_data)

    except ValueError as ve:
        print(f"PREDICT VALUE ERROR: {ve}")
        print(traceback.format_exc()) # Print full traceback
        return jsonify({"error": f"Data processing error: {ve}"}), 400
    except Exception as e:
        print(f"PREDICT UNEXPECTED ERROR for sample {data.get('sample_filename', 'N/A') if 'data' in locals() else 'N/A'}: {e}")
        print(traceback.format_exc()) # Print full traceback
        error_message = f"An internal server error occurred during prediction."
        return jsonify({"error": error_message}), 500

# --- Run Application ---
if __name__ == '__main__':
    print("Attempting to run Flask app...")
    app.run(debug=True, use_reloader=False) # Added use_reloader=False for more stable debug output
    print("Flask app should be running.")

