from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import io
import base64
from PIL import Image
import logging
import os
import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# from mediapipe_service import MediaPipeService  # Removed - Perfect Corp handles face detection
try:
    from mask_generator import MaskGenerator
    mask_generator = MaskGenerator()
    MASK_GENERATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MaskGenerator not available (likely missing graphics libs): {e}")
    mask_generator = None
    MASK_GENERATOR_AVAILABLE = False

from replicate_client import ReplicateClient
from color_transform_service import ColorTransformService
from perfect_corp_service import PerfectCorpService
from utils import process_image, encode_image_base64

# Check for required environment variables
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY")
if not REPLICATE_API_KEY or REPLICATE_API_KEY == "your-replicate-api-key-here":
    logger.warning("REPLICATE_API_KEY not set in .env file. Image generation will not work.")

app = FastAPI(title="Beauty Guide Backend", version="1.0.0")

# Create uploads directory if it doesn't exist
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Mount static files to serve uploaded images
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

# Configure CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React Native app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
# mediapipe_service = MediaPipeService()  # Removed - Perfect Corp handles face detection
# mask_generator already initialized above conditionally
replicate_client = ReplicateClient()
color_transform_service = ColorTransformService()

# Initialize Perfect Corp service with credentials from README
PERFECT_CORP_API_KEY = "sk-uRsxdXHx6gluQJYHRUOKqQRxlv9c2znmbMVmze3s6HAHLCGjr2UP-TDG-VzEqcT0"
PERFECT_CORP_SECRET_KEY = "MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQDMjLQL0qPOfPLLWAWHhkegp93WhgcR1FwZcJQBWHqPSZTN23CMQ12KLS7oukmN5VYn3EiqZ+q2efG6CdCiLS52KZffio0aQchRHJdFcIz2UVF1vgA1V1ug9pHWoBGPVOEQLNwy3xddKce8E2xQbyNLbAu73IOAzuO8yStOHe4PzwIDAQAB"
perfect_corp_service = PerfectCorpService(PERFECT_CORP_API_KEY, PERFECT_CORP_SECRET_KEY)

def save_uploaded_file(image_data: bytes, original_filename: str) -> str:
    """Save uploaded file and return public URL"""
    # Generate unique filename
    file_extension = os.path.splitext(original_filename)[1] or '.jpg'
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(UPLOADS_DIR, unique_filename)
    
    # Save file
    with open(file_path, 'wb') as f:
        f.write(image_data)
    
    # Return public URL (assuming server runs on localhost:8000)
    public_url = f"http://localhost:8000/uploads/{unique_filename}"
    logger.info(f"Saved file as {file_path}, public URL: {public_url}")
    return public_url

# Pydantic models for request/response
class MakeupSuggestion(BaseModel):
    text: str

class PerfectCorpEffect(BaseModel):
    category: str  # blush, lip_color, eye_liner, eyeshadow
    color: str     # hex code like #FF7F50
    texture: str   # matte, satin, gloss, shimmer, metallic, sheer
    intensity: int # 1-100
    placement: Optional[str] = None  # optional placement info

class MakeupAnalysis(BaseModel):
    generalFeedback: str
    specificSuggestions: List[str]
    productRecommendations: List[str]
    techniques: List[str]
    followUpQuestion: str
    perfectCorpEffects: Optional[List[PerfectCorpEffect]] = None  # Backend-only structured data

class GenerateAfterPhotoRequest(BaseModel):
    analysis: MakeupAnalysis

class GenerateAfterPhotoResponse(BaseModel):
    success: bool
    generated_image_url: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic functionality
        import time
        return {
            "status": "healthy", 
            "message": "Beauty Guide Backend is running",
            "timestamp": time.time(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/analyze-landmarks", response_model=Dict)
async def analyze_landmarks(file: UploadFile = File(...)):
    """
    Analyze facial landmarks from uploaded image
    Returns landmark coordinates for debugging/testing
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = process_image(image_data)
        
        # Extract landmarks
        landmarks = mediapipe_service.detect_face_landmarks(image)
        
        if landmarks is None:
            raise HTTPException(status_code=422, detail="No face detected in image")
        
        return {
            "success": True,
            "landmarks_count": len(landmarks),
            "image_dimensions": {"width": image.shape[1], "height": image.shape[0]},
            "landmarks": landmarks  # First 10 landmarks for preview
        }
        
    except Exception as e:
        logger.error(f"Error analyzing landmarks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/generate-mask", response_model=Dict)
async def generate_mask(
    file: UploadFile = File(...),
    target_regions: str = Form(...)  # JSON string of target regions
):
    """
    Generate mask from image and target regions
    For testing mask generation
    """
    try:
        import json
        
        # Parse target regions
        regions = json.loads(target_regions)
        
        # Validate file
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Process image
        image_data = await file.read()
        image = process_image(image_data)
        
        # Extract landmarks
        landmarks = mediapipe_service.detect_face_landmarks(image)
        if landmarks is None:
            raise HTTPException(status_code=422, detail="No face detected in image")
        
        # Generate mask
        mask_base64 = mask_generator.create_mask_from_regions(
            landmarks, regions, image.shape[1], image.shape[0]
        )
        
        return {
            "success": True,
            "mask_base64": mask_base64,
            "target_regions": regions
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid target_regions JSON")
    except Exception as e:
        logger.error(f"Error generating mask: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating mask: {str(e)}")

@app.post("/generate-after-photo", response_model=GenerateAfterPhotoResponse)
async def generate_after_photo(
    file: UploadFile = File(...),
    analysis: str = Form(...)  # JSON string of MakeupAnalysis
):
    """
    Complete pipeline: analyze image -> generate mask -> create after photo
    Main endpoint for React Native app
    """
    import time
    import json
    
    start_time = time.time()
    
    try:
        # Note: Using Perfect Corp API for professional makeup application
        logger.info("Using Perfect Corp API for professional makeup enhancement")
        
        # Parse analysis
        logger.info(f"Received analysis data: {analysis[:200]}...")
        analysis_data = json.loads(analysis)
        logger.info(f"Parsed analysis data keys: {list(analysis_data.keys())}")
        makeup_analysis = MakeupAnalysis(**analysis_data)
        logger.info("Analysis data validation successful")
        
        # Validate file
        if not file.content_type or not file.content_type.startswith("image/"):
            logger.error(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="File must be an image")
        
        logger.info(f"Starting after photo generation pipeline with file: {file.filename}, type: {file.content_type}")
        
        # Step 1: Process image
        image_data = await file.read()
        logger.info(f"Read {len(image_data)} bytes of image data")
        image = process_image(image_data)
        logger.info(f"Processed image shape: {image.shape}")
        
        logger.info("Image processed successfully")
        
        # Step 2: Apply Perfect Corp AI-driven makeup (handles face detection internally)
        logger.info("Applying Perfect Corp makeup based on AI analysis")
        
        result = perfect_corp_service.apply_ai_driven_makeup(
            image_data,
            makeup_analysis
        )
        
        processing_time = time.time() - start_time
        
        if result["success"]:
            logger.info(f"After photo generated successfully in {processing_time:.2f}s")
            return GenerateAfterPhotoResponse(
                success=True,
                generated_image_url=result["generated_image_url"],
                processing_time=processing_time
            )
        else:
            logger.error(f"Perfect Corp API error: {result['error']}")
            return GenerateAfterPhotoResponse(
                success=False,
                error=result["error"],
                processing_time=processing_time
            )
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid analysis JSON")
    except Exception as e:
        logger.error(f"Error in generate_after_photo: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return GenerateAfterPhotoResponse(
            success=False,
            error=f"Internal server error: {str(e)}",
            processing_time=time.time() - start_time
        )

@app.get("/perfect-corp-health")
async def perfect_corp_health():
    """Check Perfect Corp API connectivity"""
    try:
        result = perfect_corp_service.health_check()
        return result
    except Exception as e:
        logger.error(f"Error checking Perfect Corp health: {str(e)}")
        return {"success": False, "error": f"Health check failed: {str(e)}"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Beauty Guide Backend API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "analyze_landmarks": "/analyze-landmarks",
            "generate_mask": "/generate-mask", 
            "generate_after_photo": "/generate-after-photo",
            "perfect_corp_health": "/perfect-corp-health"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")