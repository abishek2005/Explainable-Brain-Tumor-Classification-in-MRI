# Explainable Brain Tumor Classification in MRI using Deep Learning and Grad-CAM

## Overview

This project is an AI-powered web application that classifies brain tumors from MRI scans using a deep learning model and provides visual explanations using Grad-CAM (Gradient-weighted Class Activation Mapping).

The system helps users understand why the model made a particular prediction by highlighting the important regions of the MRI image that influenced the decision.

## Features

* Brain MRI tumor classification
* Deep learning-based prediction
* Explainable AI using Grad-CAM
* Interactive web interface using Flask
* Upload MRI images for analysis
* Visual heatmap generation for model interpretation
* Supports multiple tumor categories

## Technologies Used

* Python
* TensorFlow / Keras
* Flask
* OpenCV
* NumPy
* Matplotlib
* HTML/CSS
* Grad-CAM

## Project Structure

```text
project/
│
├── app.py
├── gradcam.py
├── requirements.txt
│
├── model/
│   ├── brain_tumor_model.keras
│   └── class_names.json
│
├── templates/
│   └── index.html
│
├── static/
│   ├── uploads/
│   └── results/
│
└── README.md
```

## Dataset

The model was trained on a brain MRI dataset containing different categories of brain tumors.

Classes include:

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor

## How Grad-CAM Works

Grad-CAM provides visual explanations by highlighting regions of the MRI image that contribute most to the model's prediction.

Benefits:

* Improves model transparency
* Enhances trust in predictions
* Helps medical professionals interpret AI decisions

## Installation

### Clone Repository

```bash
git clone https://github.com/abishek2005/Explainable-Brain-Tumor-Classification-in-MRI.git
cd Explainable-Brain-Tumor-Classification-in-MRI
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Usage

1. Launch the application.
2. Upload a brain MRI image.
3. The model predicts the tumor category.
4. Grad-CAM generates a visual explanation.
5. View the prediction and highlighted MRI regions.

## Results

The model successfully classifies brain MRI images and provides interpretable visual explanations using Grad-CAM, making the prediction process more transparent and understandable.

## Future Improvements

* Higher accuracy through advanced architectures
* Multi-class confidence visualization
* Deployment on cloud platforms
* Mobile application integration
* Real-time MRI analysis

## Author

**Abishek**

B.Tech Artificial Intelligence & Machine Learning

## License

This project is developed for educational and research purposes.
