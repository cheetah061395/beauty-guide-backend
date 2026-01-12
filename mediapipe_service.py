import mediapipe as mp
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MediaPipeService:
    """Service for MediaPipe face landmark detection"""
    
    def __init__(self):
        """Initialize MediaPipe Face Mesh solution"""
        import os
        
        # Get the path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'face_landmarker.task')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model file not found at {model_path}")
        
        # Use the new MediaPipe framework with the downloaded model
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        
        logger.info(f"MediaPipe Face Mesh initialized successfully with model: {model_path}")
    
    def detect_face_landmarks(self, image: np.ndarray) -> Optional[List[Dict]]:
        """
        Detect facial landmarks from image
        
        Args:
            image: numpy array of image (BGR format from OpenCV)
            
        Returns:
            List of landmark dictionaries with x, y, z coordinates
            Returns None if no face is detected
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Process the image and detect landmarks
            results = self.face_detector.detect(mp_image)
            
            if not results.face_landmarks:
                logger.warning("No face landmarks detected")
                return None
            
            # Extract landmarks from the first (and only) detected face
            face_landmarks = results.face_landmarks[0]
            
            # Convert to normalized coordinates (0-1)
            landmarks = []
            image_height, image_width = image.shape[:2]
            
            for landmark in face_landmarks:
                landmarks.append({
                    'x': landmark.x,  # Normalized (0-1)
                    'y': landmark.y,  # Normalized (0-1) 
                    'z': landmark.z,  # Relative depth
                    'pixel_x': int(landmark.x * image_width),   # Absolute pixel coordinates
                    'pixel_y': int(landmark.y * image_height)
                })
            
            logger.info(f"Successfully detected {len(landmarks)} facial landmarks")
            return landmarks
            
        except Exception as e:
            logger.error(f"Error detecting landmarks: {str(e)}")
            return None
    
    def get_landmark_by_index(self, landmarks: List[Dict], index: int) -> Optional[Dict]:
        """Get specific landmark by index"""
        if 0 <= index < len(landmarks):
            return landmarks[index]
        return None
    
    def get_landmarks_by_indices(self, landmarks: List[Dict], indices: List[int]) -> List[Dict]:
        """Get multiple landmarks by their indices"""
        return [landmarks[i] for i in indices if 0 <= i < len(landmarks)]
    
    def get_region_landmarks(self, landmarks: List[Dict], region: str) -> List[Dict]:
        """
        Get landmarks for specific facial regions
        
        Args:
            landmarks: List of all facial landmarks
            region: Region name ('lips', 'left_eye', 'right_eye', etc.)
            
        Returns:
            List of landmarks for the specified region
        """
        # MediaPipe Face Mesh landmark indices for different regions
        region_indices = {
            # Lip landmarks (inner and outer contour)
            'lips': [
                # Outer lip contour
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
                # Inner lip contour
                78, 95, 88, 178, 87, 14, 317, 402, 318, 324
            ],
            
            # Left eye (from viewer's perspective - person's right eye)
            'left_eye': [
                # Eye contour and eyelids
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                # Upper eyelid
                246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,
                # Lower eyelid
                33, 130, 25, 110, 24, 23, 22, 26, 112, 243
            ],
            
            # Right eye (from viewer's perspective - person's left eye)  
            'right_eye': [
                # Eye contour and eyelids
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
                # Upper eyelid
                398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362,
                # Lower eyelid
                362, 359, 255, 339, 254, 253, 252, 256, 341, 463
            ],
            
            # Eyebrows
            'left_eyebrow': [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305],
            'right_eyebrow': [276, 283, 282, 281, 278, 344, 360, 363, 331, 279, 440, 75],
            
            # Cheeks (approximate areas)
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147],
            'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376],
            
            # Forehead
            'forehead': [10, 151, 9, 8, 107, 55, 8, 9, 151, 337, 299, 333, 298, 301],
            
            # Jawline and chin
            'jawline': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323],
            
            # Under-eye areas
            'left_under_eye': [35, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122],
            'right_under_eye': [261, 265, 448, 449, 450, 451, 452, 453, 464, 465, 351]
        }
        
        if region not in region_indices:
            logger.warning(f"Unknown region: {region}")
            return []
        
        indices = region_indices[region]
        return self.get_landmarks_by_indices(landmarks, indices)
    
    def calculate_face_bounds(self, landmarks: List[Dict]) -> Dict[str, int]:
        """
        Calculate bounding box of the face
        
        Returns:
            Dictionary with min_x, max_x, min_y, max_y in pixel coordinates
        """
        if not landmarks:
            return {}
        
        pixel_xs = [lm['pixel_x'] for lm in landmarks]
        pixel_ys = [lm['pixel_y'] for lm in landmarks]
        
        return {
            'min_x': min(pixel_xs),
            'max_x': max(pixel_xs),
            'min_y': min(pixel_ys),
            'max_y': max(pixel_ys),
            'width': max(pixel_xs) - min(pixel_xs),
            'height': max(pixel_ys) - min(pixel_ys)
        }
    
    def visualize_landmarks(self, image: np.ndarray, landmarks: List[Dict], 
                           region: Optional[str] = None) -> np.ndarray:
        """
        Draw landmarks on image for visualization/debugging
        
        Args:
            image: Original image
            landmarks: List of landmarks
            region: Optional region to highlight (if None, shows all landmarks)
            
        Returns:
            Image with landmarks drawn
        """
        vis_image = image.copy()
        
        if region:
            # Draw only specific region landmarks
            region_landmarks = self.get_region_landmarks(landmarks, region)
            for landmark in region_landmarks:
                cv2.circle(vis_image, 
                          (landmark['pixel_x'], landmark['pixel_y']), 
                          2, (0, 255, 0), -1)
        else:
            # Draw all landmarks
            for landmark in landmarks:
                cv2.circle(vis_image, 
                          (landmark['pixel_x'], landmark['pixel_y']), 
                          1, (0, 255, 0), -1)
        
        return vis_image
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()