import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def process_image(image_data: bytes, max_size: int = 1024) -> np.ndarray:
    """
    Process uploaded image data into OpenCV format
    
    Args:
        image_data: Raw image bytes
        max_size: Maximum dimension (width or height) for resizing
        
    Returns:
        OpenCV image array (BGR format)
    """
    try:
        # Debug: Log image data info
        logger.info(f"Processing image data: type={type(image_data)}, size={len(image_data) if hasattr(image_data, '__len__') else 'unknown'} bytes")
        
        # Create BytesIO and ensure it's at the beginning
        if isinstance(image_data, bytes):
            image_stream = io.BytesIO(image_data)
        else:
            # Handle case where image_data might be BytesIO already
            image_stream = image_data
            
        image_stream.seek(0)  # Ensure stream is at beginning
        
        # Try to detect image format first
        image_header = image_stream.read(12)
        image_stream.seek(0)  # Reset stream
        
        logger.info(f"Image header (first 12 bytes): {image_header.hex() if len(image_header) >= 4 else 'too short'}")
        
        # Convert bytes to PIL Image
        image = Image.open(image_stream)
        
        # Log image info
        logger.info(f"PIL Image loaded: format={image.format}, mode={image.mode}, size={image.size}")
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.info(f"Converted image mode to RGB")
        
        # Resize if too large (maintaining aspect ratio)
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int((height * max_size) / width)
            else:
                new_height = max_size
                new_width = int((width * max_size) / height)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Convert PIL to OpenCV format (RGB to BGR)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        logger.info(f"Successfully processed image: {opencv_image.shape}")
        return opencv_image
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"Invalid image data: {str(e)}")

def encode_image_base64(image_data: bytes, format: str = "JPEG") -> str:
    """
    Encode image bytes to base64 data URL
    
    Args:
        image_data: Raw image bytes
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        Base64 data URL string
    """
    try:
        # Process through PIL to ensure proper format
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB for JPEG format
        if format.upper() == "JPEG" and image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format=format.upper())
        buffer.seek(0)
        
        # Encode to base64
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create data URL
        mime_type = f"image/{format.lower()}"
        data_url = f"data:{mime_type};base64,{base64_string}"
        
        return data_url
        
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        raise ValueError(f"Failed to encode image: {str(e)}")

def encode_opencv_image_base64(opencv_image: np.ndarray, format: str = "PNG") -> str:
    """
    Encode OpenCV image array to base64 data URL
    
    Args:
        opencv_image: OpenCV image array (BGR format)
        format: Output format (PNG, JPEG, etc.)
        
    Returns:
        Base64 data URL string
    """
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_image = Image.fromarray(rgb_image)
        
        # Convert to RGB for JPEG format
        if format.upper() == "JPEG" and pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format.upper())
        buffer.seek(0)
        
        # Encode to base64
        base64_string = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create data URL
        mime_type = f"image/{format.lower()}"
        data_url = f"data:{mime_type};base64,{base64_string}"
        
        return data_url
        
    except Exception as e:
        logger.error(f"Error encoding OpenCV image to base64: {str(e)}")
        raise ValueError(f"Failed to encode OpenCV image: {str(e)}")

def decode_base64_image(base64_string: str) -> Tuple[np.ndarray, str]:
    """
    Decode base64 image string to OpenCV format
    
    Args:
        base64_string: Base64 image string (with or without data URL prefix)
        
    Returns:
        Tuple of (opencv_image, original_format)
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image/'):
            header, base64_data = base64_string.split(',', 1)
            # Extract format from header
            format_part = header.split(';')[0].split('/')[1]
        else:
            base64_data = base64_string
            format_part = 'unknown'
        
        # Decode base64
        image_bytes = base64.b64decode(base64_data)
        
        # Convert to PIL
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return opencv_image, format_part
        
    except Exception as e:
        logger.error(f"Error decoding base64 image: {str(e)}")
        raise ValueError(f"Failed to decode base64 image: {str(e)}")

def validate_image_dimensions(image: np.ndarray, min_size: int = 64, max_size: int = 2048) -> bool:
    """
    Validate image dimensions are within acceptable range
    
    Args:
        image: OpenCV image array
        min_size: Minimum acceptable dimension
        max_size: Maximum acceptable dimension
        
    Returns:
        True if dimensions are valid
    """
    height, width = image.shape[:2]
    
    if width < min_size or height < min_size:
        logger.warning(f"Image too small: {width}x{height}, minimum: {min_size}")
        return False
    
    if width > max_size or height > max_size:
        logger.warning(f"Image too large: {width}x{height}, maximum: {max_size}")
        return False
    
    return True

def resize_image_maintain_aspect(image: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize image maintaining aspect ratio
    
    Args:
        image: OpenCV image array
        target_size: Target size for the larger dimension
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if max(width, height) <= target_size:
        return image  # No resize needed
    
    # Calculate new dimensions
    if width > height:
        new_width = target_size
        new_height = int((height * target_size) / width)
    else:
        new_height = target_size
        new_width = int((width * target_size) / height)
    
    # Resize using high-quality interpolation
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    return resized

def create_debug_image(image: np.ndarray, landmarks: list, title: str = "Debug") -> str:
    """
    Create debug image with landmarks overlaid
    
    Args:
        image: Original OpenCV image
        landmarks: List of landmark dictionaries
        title: Title for the debug image
        
    Returns:
        Base64 encoded debug image
    """
    try:
        debug_image = image.copy()
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            x, y = landmark.get('pixel_x', 0), landmark.get('pixel_y', 0)
            cv2.circle(debug_image, (x, y), 2, (0, 255, 0), -1)
            
            # Draw index for first 20 landmarks
            if i < 20:
                cv2.putText(debug_image, str(i), (x+3, y-3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Add title
        cv2.putText(debug_image, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Convert to base64
        return encode_opencv_image_base64(debug_image, "PNG")
        
    except Exception as e:
        logger.error(f"Error creating debug image: {str(e)}")
        return ""

def calculate_image_quality_score(image: np.ndarray) -> float:
    """
    Calculate a simple quality score for the image
    
    Args:
        image: OpenCV image array
        
    Returns:
        Quality score between 0 and 1 (higher is better)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (higher variance = sharper image)
        # Values above 100 are considered good quality
        sharpness_score = min(laplacian_var / 100.0, 1.0)
        
        # Calculate brightness distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        
        # Good images have more uniform brightness distribution
        brightness_score = 1.0 - np.sum(hist_norm ** 2)  # Inverse of concentration
        
        # Combine scores
        quality_score = (sharpness_score * 0.7 + brightness_score * 0.3)
        
        logger.debug(f"Image quality score: {quality_score:.3f} (sharpness: {sharpness_score:.3f}, brightness: {brightness_score:.3f})")
        
        return quality_score
        
    except Exception as e:
        logger.error(f"Error calculating image quality: {str(e)}")
        return 0.5  # Default medium quality score

def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    
    # Remove or replace problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove any remaining non-ASCII characters
    safe_name = re.sub(r'[^\x00-\x7F]+', '_', safe_name)
    
    # Limit length
    if len(safe_name) > 100:
        name, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
        safe_name = name[:95] + ('.' + ext if ext else '')
    
    return safe_name

def log_performance(func_name: str, start_time: float, end_time: float, 
                   additional_info: str = "") -> None:
    """
    Log performance metrics for debugging
    
    Args:
        func_name: Name of the function being measured
        start_time: Start timestamp
        end_time: End timestamp
        additional_info: Additional context information
    """
    duration = end_time - start_time
    
    if additional_info:
        logger.info(f"Performance: {func_name} took {duration:.3f}s - {additional_info}")
    else:
        logger.info(f"Performance: {func_name} took {duration:.3f}s")