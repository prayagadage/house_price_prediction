# 🏠 House Rent Prediction App

A full-stack Machine Learning application that predicts house rent prices based on features like BHK, Size, City, and Furnishing status.

![DreamHome UI](https://via.placeholder.com/800x400?text=DreamHome+Rent+Predictor+UI)

## ✨ Features
- **Accurate Predictions**: Uses a Ridge Regression model trained on real estate data.
- **Interactive UI**: A beautiful, glassmorphism-styled frontend for easy data entry.
- **Real-time API**: Fast, asynchronous backend powered by FastAPI.
- **Responsive Design**: Works on desktop and mobile.

## 🛠️ Tech Stack
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript
- **Backend**: Python, FastAPI, Uvicorn
- **Machine Learning**: Scikit-Learn, Pandas, NumPy
- **Deployment**: Render

## 📂 Project Structure
```text
house_price_prediction/
├── backend/                  # Backend Logic
│   ├── core/                 # ML Source Code
│   │   ├── predict.py        # Inference logic
│   │   ├── train.py          # Training pipeline
│   │   └── data_preprocessing.py
│   ├── models/               # Saved ML Models (.pkl)
│   ├── data/                 # Raw Dataset
│   └── main.py               # FastAPI Entry Point (Server)
├── frontend/                 # Web Interface
│   ├── index.html            # Main User Interface
│   ├── style.css             # Styling
│   └── script.js             # API Integration
├── notebooks/                # Jupyter Notebooks for EDA
├── requirements.txt          # Python Dependencies
└── render.yaml               # Deployment Configuration
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/prayagadage/house_price_prediction.git
cd house_price_prediction
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App Locally
Start the unified server (Frontend + Backend):
```bash
cd backend
uvicorn main:app --reload --port 8000
```
Open your browser and visit: **http://localhost:8000**

## 🧠 Model Training (Optional)
If you want to retrain the model with new data:
```bash
python backend/core/train.py
```
This will generate a new `rent_prediction_model.pkl` in the `backend/models` directory.

## 🌐 API Endpoints
- `GET /`: Serves the Frontend.
- `POST /predict`: Accepts house features and returns predicted rent.

**Example Request:**
```json
{
  "BHK": 2,
  "Size": 900,
  "Bathroom": 2,
  "current_floor": 2,
  "total_floors": 5,
  "Area Type": "Super Area",
  "Area Locality": "Whitefield",
  "City": "Bangalore",
  "Furnishing Status": "Semi-Furnished",
  "Tenant Preferred": "Bachelors/Family"
}
```

## ☁️ Deployment
This app is configured for deployment on **Render**.
1. Push to GitHub.
2. Create a new **Web Service** on Render.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

---
Made with ❤️ by Prayag Adage