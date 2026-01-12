from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Beauty Guide Backend", version="1.0.0")

# Configure CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model for image generation
class GenerateAfterPhotoResponse(BaseModel):
    success: bool
    generated_image_url: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Beauty Guide Backend is running"}

@app.post("/generate-after-photo", response_model=GenerateAfterPhotoResponse)
async def generate_after_photo_temp(
    file: UploadFile = File(...),
    analysis: str = Form(...)
):
    """
    Temporary endpoint that provides user feedback while MediaPipe is being fixed
    """
    import time
    start_time = time.time()
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            return GenerateAfterPhotoResponse(
                success=False,
                error="File must be an image",
                processing_time=time.time() - start_time
            )
        
        # Read file to ensure it's valid
        file_data = await file.read()
        file_size = len(file_data)
        
        logger.info(f"Received image upload: {file.filename}, size: {file_size} bytes")
        logger.info(f"Analysis data received: {len(analysis)} characters")
        
        # Return helpful message about current status
        return GenerateAfterPhotoResponse(
            success=False,
            error="Image generation is temporarily unavailable while we update the face detection system. Please try again in a few minutes.",
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Error in temporary endpoint: {str(e)}")
        return GenerateAfterPhotoResponse(
            success=False,
            error="Unable to process image at this time. Please try again later.",
            processing_time=time.time() - start_time
        )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Beauty Guide Backend API",
        "version": "1.0.0",
        "status": "MediaPipe being updated - image generation temporarily unavailable",
        "endpoints": {
            "health": "/health",
            "generate_after_photo": "/generate-after-photo (temporary)"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")