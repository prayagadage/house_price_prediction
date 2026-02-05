from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

# Add the project root directory to sys.path
# This allows importing 'src' regardless of where the script is run from
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_rent

app = FastAPI(title="House Rent Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity (or specify frontend URL)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HouseFeatures(BaseModel):
    BHK: int
    Size: int
    Bathroom: int
    current_floor: int
    total_floors: int
    Area_Type: str
    Area_Locality: str
    City: str
    Furnishing_Status: str
    Tenant_Preferred: str

    class Config:
        populate_by_name = True

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        # Convert Pydantic model to dict, mapping keys to match training data
        data = {
            "BHK": features.BHK,
            "Size": features.Size,
            "Bathroom": features.Bathroom,
            "current_floor": features.current_floor,
            "total_floors": features.total_floors,
            "Area Type": features.Area_Type,
            "Area Locality": features.Area_Locality,
            "City": features.City,
            "Furnishing Status": features.Furnishing_Status,
            "Tenant Preferred": features.Tenant_Preferred
        }
        
        prediction = predict_rent(data)
        return {"predicted_rent": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount Frontend (Static Files)
# This must be after API routes so it doesn't override them
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
