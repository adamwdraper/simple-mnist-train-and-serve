import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import numpy as np
import io
import base64 # For potentially handling base64 encoded images later
from fastapi.staticfiles import StaticFiles # For static files if needed later
from fastapi.templating import Jinja2Templates # For serving HTML
from fastapi import Request # To use in HTML rendering
from model import SimpleNN # <--- IMPORT ADDED
import os # For path joining

# --- Model Definition (Copied from train.py) --- REMOVED
# class SimpleNN(nn.Module):
#    ...

# --- Configuration ---
MODEL_PATH = "mnist_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model and Transforms ---
model = SimpleNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Set model to evaluation mode

# Image transformations (must be same as during training)
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --- FastAPI Application ---
app = FastAPI(title="MNIST Digit Recognizer API")

# Configure Jinja2 templates
# Determine the absolute path to the templates directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Serve static files (if you add separate CSS/JS files later)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Input/Output Models ---
class ImageData(BaseModel):
    # Option 1: Expect a flat list of 784 pixel values (0-255)
    pixels: list[float] # Could also be int if you ensure 0-255 range
    # Option 2: Expect a base64 encoded image string (more flexible)
    # image_base64: str | None = None 

class Prediction(BaseModel):
    predicted_digit: int
    scores: list[float] # Raw scores from the model for each digit

# --- API Endpoints ---
@app.get("/")
async def read_root_ui(request: Request):
    """Serves the main UI page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docs-api") # Keep original /docs for FastAPI's own docs
def read_api_docs_message():
    return {"message": "API specific documentation is typically found at /docs or /redoc provided by FastAPI itself."}

@app.post("/predict", response_model=Prediction)
def predict_digit(image_data: ImageData):
    """
    Predicts the digit from a list of pixel values.
    The input `pixels` should be a flat list of 784 values (28x28 image),
    normalized to be between 0 and 1 (or 0-255, which will be scaled).
    """
    try:
        # Assuming pixels are 0-255, convert to numpy array and then PIL Image
        # If pixels are already normalized (0-1), this part needs adjustment
        # For now, let's assume they are 0-255 for easier input, then scale.
        if not (len(image_data.pixels) == 28 * 28):
             raise ValueError(f"Input pixel list must have {28*28} values, got {len(image_data.pixels)}")
        
        # Scale if pixels are 0-255, ensure they are float for PIL
        pixel_array_np = np.array(image_data.pixels, dtype=np.float32)
        if np.max(pixel_array_np) > 1.0: # Heuristic to check if normalization is needed
            pixel_array_np = pixel_array_np / 255.0
        
        pixel_array_np = pixel_array_np.reshape(28, 28)

        # Convert numpy array to PIL Image (single channel, grayscale)
        # PIL expects uint8 for mode 'L', so we scale to 0-255 if it was 0-1, then convert
        # For ToTensor(), it's better to keep it as float and let ToTensor handle it.
        # So, if the input pixels were 0-255, we normalized to 0-1 above. This is fine for ToTensor.
        image = Image.fromarray(pixel_array_np.astype(np.float32), mode='F') # 'F' for float32

        # Apply transformations
        tensor = img_transform(image).unsqueeze(0).to(device) # Add batch dimension

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted_idx = torch.max(outputs.data, 1)
            probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()

        return Prediction(predicted_digit=predicted_idx.item(), scores=probabilities)

    except ValueError as ve:
        return {"error": str(ve), "status_code": 400} # Or raise HTTPException
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during prediction: {e}")
        return {"error": "Error processing the image.", "status_code": 500} # Or raise HTTPException

# --- To Run the Server (from terminal) ---
# uvicorn serve:app --reload
# Then go to http://127.0.0.1:8000 to see the UI
# API docs at http://127.0.0.1:8000/docs or http://127.0.0.1:8000/redoc 