# NephroScan AI: Kidney Stone Ultrasound Detection & Diagnostic Suite

Deep learning medical imaging pipeline and interactive clinical dashboard to detect nephrolithiasis (kidney stones) in B-mode ultrasound scans using Convolutional Neural Networks (MobileNetV2 Transfer Learning), Explainable AI (Grad-CAM), and Out-Of-Distribution (OOD) modality validation.

---

## 🩺 System Architecture & Pipeline

```
kidney-stone-detection-aiml/
├── app/
│   └── streamlit_app.py        # Clinical Streamlit Web App (Grad-CAM, Pelvicalyceal Dilation)
├── models/
│   ├── best_model.h5           # High-accuracy MobileNetV2 checkpoint (98.87% accuracy)
│   ├── confusion_matrix_eval.png
│   └── evaluation_report.txt
├── src/
│   ├── __init__.py             # Python ML package exports
│   ├── config.py               # Centralized hyperparameters, paths & modality thresholds
│   ├── modality_guard.py       # Out-Of-Distribution (OOD) guard (rejects non-ultrasound photos)
│   ├── gradcam.py              # Grad-CAM heatmap generator for focal calculus localization
│   ├── predict_image.py        # Core inference engine & clinical advisory logic
│   ├── preprocess.py           # Automated dataset partitioning & data generators
│   ├── train_cnn.py            # Transfer learning training loop with early stopping
│   ├── evaluate.py             # Metric calculation (Accuracy, F1, Precision, Recall)
│   └── data_cleaning.py        # Dataset corruption scan & duplicate detection utility
├── data/                       # Local train/val/test splits (gitignored)
├── raw_dataset/                # Local original dataset (gitignored)
├── requirements.txt            # Python dependencies
├── run_pipeline.py             # End-to-end pipeline execution script
└── .gitignore                  # Full protection preventing images & weights leaking to GitHub
```

---

## 🚀 Key Improvements & Features

1. **Out-of-Distribution Modality Guard (`src/modality_guard.py`)**:
   - Closed-world binary CNNs erroneously predict 100% on random non-medical images (e.g. photos of animals, cars, landscapes).
   - The modality guard inspects chromaticity saturation, RGB channel divergence, dynamic contrast, and speckle noise variance to reject non-ultrasound inputs prior to inference.

2. **Explainable AI with Grad-CAM (`src/gradcam.py`)**:
   - Visualizes the convolutional activation map (`out_relu` layer) directly onto the ultrasound scan.
   - Highlights the exact hyperechoic calculus focus and posterior acoustic shadowing region.

3. **Clinical Pelvicalyceal Dilation & Severity Index**:
   - Estimates obstruction severity, hydronephrosis risk (Low, Moderate, Elevated), and outputs evidence-based urological advisory recommendations.

4. **Zero-Leakage Git Protection**:
   - `.gitignore` strictly protects datasets (`raw_dataset/`, `data/`, `*.jpg`, `*.png`, `*.npy`) from accidental commits to GitHub while keeping all images safe on your local drive.

---

## 🛠️ Installation & Setup

Ensure Python 3.10 or 3.11 is installed (TensorFlow 2.x supports Python $\le$ 3.11):

```bash
# 1. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install requirements
pip install -r requirements.txt
```

---

## 💻 How to Run

### 1. Launch the Clinical Web Application
```bash
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 2. Run Inference on a Single Image via CLI
```bash
python src/predict_image.py path/to/ultrasound.jpg
```

### 3. Evaluate Model on Test Dataset
```bash
python src/evaluate.py
```
*Current benchmark on 1,676 test scans:*
- **Accuracy**: 98.87%
- **Weighted Precision**: 0.99
- **Weighted Recall**: 0.98
- **Weighted F1-Score**: 0.99

### 4. Re-train the CNN Architecture
```bash
python src/train_cnn.py
```

### 5. Run the Entire Pipeline Automatically
```bash
python run_pipeline.py
```

---

## 🔒 Privacy & Data Safety

All dataset images are strictly retained on your local system and excluded from Git tracking via `.gitignore`. No medical imaging data is uploaded to remote repositories.
