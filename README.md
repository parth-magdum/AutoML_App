# 🚀 FastAPI + PyCaret AutoML App

This project is a web-based **AutoML app** built with **FastAPI**, allowing users to:
- Upload any CSV dataset 📂
- Specify the target column 🎯
- Automatically run **PyCaret** to train & compare models 🧠
- See predictions and download them as a CSV ✅

Ideal for quickly testing machine learning on any tabular data without writing code!

---

## 🛠 Prerequisites

- Python 3.10 (PyCaret does **not** support Python 3.12+)
- `pip` installed

To check Python version:

```bash
py --version
```

---

## 🔧 Setup Instructions (Windows)

### 1.  Create & activate a virtual environment in Python 3.10

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```


### 2.  Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3.  Run the FastAPI app

```bash
uvicorn app.main:app --reload --port 8000
```

Then open your browser and go to:  
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📋 How It Works

1. Upload a CSV file (with headers).
2. Enter the **target column** name (the label you want to predict).
3. The app:
   - Loads your dataset using Pandas
   - Sets up a PyCaret classification pipeline
   - Compares multiple models using `compare_models()`
   - Shows the prediction results in a table
   - Lets you download the prediction CSV
