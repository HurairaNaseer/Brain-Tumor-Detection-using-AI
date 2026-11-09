# 🧠 Brain Tumor Detection using AI

An end-to-end Medical AI system that detects and classifies brain tumors from MRI scans using Deep Learning. Upload an MRI scan, and the AI analyzes it and returns the tumor type along with a confidence score.


## 🔬 Overview

Brain tumors are among the most critical medical conditions, and early detection greatly improves treatment outcomes. This project provides a complete pipeline:

1. **Model training** — a deep learning model trained on MRI images using transfer learning.
2. **Backend** — a FastAPI REST service that serves model predictions.
3. **Frontend** — a clean Streamlit web app where users upload an MRI scan and see the results instantly.

**Workflow:**

```
Upload MRI Scan  →  AI Analysis  →  Tumor Type + Confidence Score
```

---

## 🚀 Key Features

- ⚡ **Real-time MRI analysis**
- 📊 **Confidence scores** displayed with visual bars for every class
- 🔌 **REST API** built with FastAPI
- 🎨 **Clean & professional UI** built with Streamlit
- 🧩 **Modular design** — frontend and backend are decoupled

---

## ⚙️ Tech Stack

| Layer            | Technology                          |
| ---------------- | ----------------------------------- |
| Model            | EfficientNetB3 (Transfer Learning)  |
| Deep Learning    | TensorFlow & Keras                  |
| Backend          | FastAPI                             |
| Frontend         | Streamlit                           |
| Language         | Python                              |

---

## 📊 Model Details

| Property             | Value                                              |
| -------------------- | -------------------------------------------------- |
| Architecture         | EfficientNetB3 (pre-trained, fine-tuned)           |
| Dataset size         | 12,841 MRI images                                  |
| Validation accuracy  | **92%**                                            |
| Number of classes    | 4                                                  |
| Classes              | Glioma, Meningioma, Pituitary, No Tumor            |

### Classes

- **Glioma** — tumor arising from glial cells of the brain or spine
- **Meningioma** — tumor arising from the meninges (membranes surrounding the brain)
- **Pituitary** — tumor located in the pituitary gland
- **No Tumor** — healthy MRI scan

---

## 📁 Project Structure

> Adjust this to match your actual folders.

```
brain-tumor-detection/
│
├── backend/
│   ├── main.py              # FastAPI application
│   └── model/
│       └── model.h5         # Trained model weights
│
├── frontend/
│   └── app.py               # Streamlit application
│
├── notebooks/
│   └── training.ipynb       # Model training notebook
│
├── requirements.txt
└── README.md
```

---

## 🛠 Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

```
tensorflow
keras
fastapi
uvicorn
streamlit
pillow
numpy
requests
python-multipart
```

---

## 📥 Model Download

The trained model file is too large for GitHub, so it is hosted on Google Drive.

👉 **[Download the trained model](https://drive.google.com/file/d/19heWz8YCBV2sbdEToL7S76NwKOFmusHK/view?usp=sharing)**

After downloading, place the file in the `backend/model/` folder (create the folder if it does not exist) so the structure looks like this:

```
backend/
└── model/
    └── model.h5     # downloaded file (keep the same name used in main.py)
```

---

## ▶️ Usage

You need to run the backend and the frontend in two separate terminals.

### Start the backend (FastAPI)

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
Interactive docs (Swagger UI): `http://127.0.0.1:8000/docs`

### Start the frontend (Streamlit)

```bash
cd frontend
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Using the app

1. Open the Streamlit app.
2. Upload an MRI scan (`.jpg`, `.jpeg`, or `.png`).
3. Click **Analyze**.
4. View the predicted tumor type and the confidence score for each class.

---

## 🔌 API Reference

> Update endpoint names to match your `main.py`.

### `POST /predict`

Accepts an MRI image and returns the predicted class.

**Request:** `multipart/form-data` with an image file

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@sample_mri.jpg"
```

**Response:**

```json
{
  "prediction": "Glioma",
  "confidence": 0.94,
  "probabilities": {
    "Glioma": 0.94,
    "Meningioma": 0.03,
    "Pituitary": 0.02,
    "No Tumor": 0.01
  }
}
```

---

## 📈 Results

- **Validation accuracy:** 92%
- Trained on **12,841** MRI images across **4 classes**

*(You can add a confusion matrix, accuracy/loss curves, and UI screenshots here.)*

```
![Confusion Matrix](assets/confusion_matrix.png)
![App Screenshot](assets/screenshot.png)
```

---

## ⚠️ Limitations & Disclaimer

This project is built for **educational and research purposes only**. It is **not** a certified medical device and must **not** be used as a substitute for professional medical diagnosis. Always consult a qualified radiologist or physician for medical decisions.

---

## 🔮 Future Work

- Improve accuracy with a larger and more diverse dataset
- Add Grad-CAM heatmaps to visualize the tumor region
- Tumor segmentation in addition to classification
- Docker support and cloud deployment
- Support for DICOM files

---
