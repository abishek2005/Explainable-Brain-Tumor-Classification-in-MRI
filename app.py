import os
import json
import uuid
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
from gradcam import generate_gradcam, overlay_heatmap

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULT_FOLDER']]:
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
# ── Model loading ─────────────────────────────────────────────
def load_model_with_fallback():
    keras_path = 'model/brain_tumor_model.keras'
    h5_path    = 'model/brain_tumor_model.h5'

    if os.path.exists(keras_path):
        print("Loading .keras format...")
        m = tf.keras.models.load_model(keras_path)
        print(f"Model loaded ✓  Input: {m.input_shape}  Output: {m.output_shape}")
        return m

    elif os.path.exists(h5_path):
        print("Loading .h5 with compile=False...")
        try:
            m = tf.keras.models.load_model(h5_path, compile=False)
            print(f"Model loaded ✓  Input: {m.input_shape}  Output: {m.output_shape}")
            return m
        except Exception as e:
            raise RuntimeError(f"Failed to load .h5: {e}")

    else:
        raise FileNotFoundError(
            "No model file found in model/ folder.\n"
            "Expected: model/brain_tumor_model.keras  OR  model/brain_tumor_model.h5"
        )

model = load_model_with_fallback()

# ── Load class names ──────────────────────────────────────────
with open('model/class_names.json') as f:
    class_names = json.load(f)
print("Classes:", class_names)

# ── Preprocessing ─────────────────────────────────────────────
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', class_names=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    uid = str(uuid.uuid4())[:8]
    filename = f"{uid}_{file.filename}"
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    img_array = preprocess_image(upload_path)
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    pred_class = class_names[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    heatmap = generate_gradcam(model, img_array, pred_idx)
    result_filename = f"gradcam_{uid}.png"
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    overlay_heatmap(upload_path, heatmap, result_path)

    all_probs = {
        class_names[i]: round(float(preds[i]) * 100, 2)
        for i in range(len(class_names))
    }

    return jsonify({
        'prediction': pred_class,
        'confidence': round(confidence, 2),
        'all_probabilities': all_probs,
        'gradcam_image': f'/{result_path.replace(os.sep, "/")}',
        'explanation': get_explanation(pred_class, confidence)
    })

def get_explanation(pred_class, confidence):
    explanations = {
        'glioma':      "Grad-CAM highlights irregular tissue in the cerebral area, consistent with glioma characteristics.",
        'meningioma':  "Activation focuses on the meningeal boundary where meningiomas typically originate.",
        'pituitary':   "The model attends to the pituitary region at the base of the brain.",
        'notumor':     "No concentrated activation — attention is uniformly distributed, indicating no anomaly detected."
    }
    key = pred_class.lower().replace(' ', '').replace('_', '')
    return explanations.get(key, "Grad-CAM highlights the regions most influential in the model's decision.")

if __name__ == '__main__':
    app.run(debug=True, port=5000)