import cv2
import numpy as np
from PIL import Image, ImageDraw
import base64
import io
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MaskGenerator:
    """Service for generating precise masks from facial landmarks"""
    
    def __init__(self):
        """Initialize mask generator"""
        logger.info("Mask generator initialized")
    
    def identify_target_regions(self, makeup_analysis) -> List[Dict]:
        """
        Identify target facial regions from makeup analysis
        
        Args:
            makeup_analysis: MakeupAnalysis object with suggestions
            
        Returns:
            List of region dictionaries with name and description
        """
        regions = []
        
        # Combine all suggestions into one text for analysis
        all_suggestions = " ".join(makeup_analysis.specificSuggestions).lower()
        general_feedback = makeup_analysis.generalFeedback.lower()
        combined_text = f"{all_suggestions} {general_feedback}"
        
        # Check for lip-related suggestions
        if any(keyword in combined_text for keyword in ['lip', 'mouth', 'lipstick', 'gloss']):
            regions.append({
                'name': 'lips',
                'description': 'Lip area for color and shape enhancement'
            })
        
        # Check for eye-related suggestions
        if any(keyword in combined_text for keyword in ['eye', 'eyelid', 'lash', 'shadow', 'liner']):
            regions.append({
                'name': 'left_eye',
                'description': 'Left eye area for shadow and liner'
            })
            regions.append({
                'name': 'right_eye', 
                'description': 'Right eye area for shadow and liner'
            })
        
        # Check for eyebrow suggestions
        if any(keyword in combined_text for keyword in ['brow', 'eyebrow']):
            regions.append({
                'name': 'left_eyebrow',
                'description': 'Left eyebrow shaping and color'
            })
            regions.append({
                'name': 'right_eyebrow',
                'description': 'Right eyebrow shaping and color'
            })
        
        # Check for cheek/blush suggestions
        if any(keyword in combined_text for keyword in ['cheek', 'blush', 'contour', 'rouge']):
            regions.append({
                'name': 'left_cheek',
                'description': 'Left cheek for blush and contouring'
            })
            regions.append({
                'name': 'right_cheek',
                'description': 'Right cheek for blush and contouring'
            })
        
        # Check for skin/foundation suggestions
        if any(keyword in combined_text for keyword in ['skin', 'foundation', 'complexion', 'coverage']):
            regions.append({
                'name': 'forehead',
                'description': 'Forehead area for skin tone adjustment'
            })
        
        # Check for under-eye suggestions
        if any(keyword in combined_text for keyword in ['under eye', 'dark circle', 'concealer', 'under-eye']):
            regions.append({
                'name': 'left_under_eye',
                'description': 'Left under-eye area for concealing'
            })
            regions.append({
                'name': 'right_under_eye',
                'description': 'Right under-eye area for concealing'
            })
        
        # Check for jawline/contouring suggestions
        if any(keyword in combined_text for keyword in ['jaw', 'chin', 'sculpt', 'define', 'jawline']):
            regions.append({
                'name': 'jawline',
                'description': 'Jawline for contouring and definition'
            })
        
        logger.info(f"Identified {len(regions)} target regions: {[r['name'] for r in regions]}")
        return regions
    
    def create_mask_from_regions(self, landmarks: List[Dict], target_regions: List[Dict], 
                                image_width: int, image_height: int) -> str:
        """
        Create precise mask from facial landmarks and target regions
        
        Args:
            landmarks: List of facial landmarks with pixel coordinates
            target_regions: List of target region dictionaries
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            Base64 encoded mask image (PNG format)
        """
        try:
            # Create PIL image for mask (RGB mode for better compatibility)
            mask_image = Image.new('RGB', (image_width, image_height), 'black')
            draw = ImageDraw.Draw(mask_image)
            
            logger.info(f"Creating mask for {len(target_regions)} regions")
            
            # Process each target region
            for region in target_regions:
                region_name = region['name']
                
                # Get landmarks for this region
                region_landmarks = self._get_region_landmark_indices(region_name)
                
                if not region_landmarks:
                    logger.warning(f"No landmarks found for region: {region_name}")
                    continue
                
                # Extract pixel coordinates for this region
                region_points = []
                for landmark_idx in region_landmarks:
                    if landmark_idx < len(landmarks):
                        landmark = landmarks[landmark_idx]
                        region_points.append((landmark['pixel_x'], landmark['pixel_y']))
                
                if len(region_points) < 3:  # Need at least 3 points for a polygon
                    logger.warning(f"Not enough points for region {region_name}: {len(region_points)}")
                    continue
                
                # Draw filled polygon for this region in white
                try:
                    draw.polygon(region_points, fill='white', outline='white')
                    logger.debug(f"Drew polygon for {region_name} with {len(region_points)} points")
                except Exception as e:
                    logger.warning(f"Failed to draw polygon for {region_name}: {e}")
                    # Fallback: draw individual points as circles
                    for point in region_points:
                        draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3], 
                                   fill='white', outline='white')
            
            # Convert PIL image to base64
            buffer = io.BytesIO()
            mask_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            mask_data_url = f"data:image/png;base64,{mask_base64}"
            
            logger.info("Mask generation completed successfully")
            return mask_data_url
            
        except Exception as e:
            logger.error(f"Error creating mask: {str(e)}")
            # Return a fallback simple mask
            return self._create_fallback_mask(image_width, image_height, target_regions)
    
    def _get_region_landmark_indices(self, region_name: str) -> List[int]:
        """
        Get MediaPipe landmark indices for specific facial regions
        These indices correspond to the 468-point MediaPipe Face Mesh
        """
        region_indices = {
            # Lip landmarks (outer contour for better coverage)
            'lips': [
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82
            ],
            
            # Left eye (comprehensive coverage)
            'left_eye': [
                # Main eye contour
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                # Extended area for shadow
                130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28, 27, 29, 30
            ],
            
            # Right eye (comprehensive coverage)
            'right_eye': [
                # Main eye contour  
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
                # Extended area for shadow
                359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258, 257, 259, 260
            ],
            
            # Eyebrows (extended for better coverage)
            'left_eyebrow': [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305, 70, 63, 105, 66, 107, 55, 65],
            'right_eyebrow': [276, 283, 282, 281, 278, 344, 360, 363, 331, 279, 440, 75, 300, 293, 334, 296, 336, 285, 295],
            
            # Cheeks (expanded area)
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 123, 137, 177, 50, 36],
            'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376, 352, 366, 401, 280, 266],
            
            # Forehead 
            'forehead': [10, 151, 9, 8, 107, 55, 8, 9, 151, 337, 299, 333, 298, 301, 368, 10, 151, 337, 299, 333],
            
            # Jawline and chin
            'jawline': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284],
            
            # Under-eye areas (extended)
            'left_under_eye': [35, 31, 228, 229, 230, 231, 232, 233, 244, 245, 122, 6, 202, 214, 234],
            'right_under_eye': [261, 265, 448, 449, 450, 451, 452, 453, 464, 465, 351, 236, 422, 434, 454]
        }
        
        return region_indices.get(region_name, [])
    
    def _create_fallback_mask(self, image_width: int, image_height: int, 
                            target_regions: List[Dict]) -> str:
        """
        Create a simple fallback mask when precise landmark-based mask fails
        """
        try:
            mask_image = Image.new('RGB', (image_width, image_height), 'black')
            draw = ImageDraw.Draw(mask_image)
            
            # Simple geometric shapes for each region
            for region in target_regions:
                region_name = region['name']
                
                if 'lip' in region_name:
                    # Simple lip area
                    center_x, center_y = image_width // 2, int(image_height * 0.75)
                    width, height = int(image_width * 0.08), int(image_height * 0.04)
                    draw.ellipse([center_x - width, center_y - height, 
                                center_x + width, center_y + height], fill='white')
                
                elif 'left_eye' in region_name:
                    center_x, center_y = int(image_width * 0.35), int(image_height * 0.4)
                    width, height = int(image_width * 0.08), int(image_height * 0.05)
                    draw.ellipse([center_x - width, center_y - height,
                                center_x + width, center_y + height], fill='white')
                
                elif 'right_eye' in region_name:
                    center_x, center_y = int(image_width * 0.65), int(image_height * 0.4)
                    width, height = int(image_width * 0.08), int(image_height * 0.05)
                    draw.ellipse([center_x - width, center_y - height,
                                center_x + width, center_y + height], fill='white')
                
                elif 'cheek' in region_name:
                    if 'left' in region_name:
                        center_x, center_y = int(image_width * 0.25), int(image_height * 0.6)
                    else:
                        center_x, center_y = int(image_width * 0.75), int(image_height * 0.6)
                    width, height = int(image_width * 0.06), int(image_height * 0.08)
                    draw.ellipse([center_x - width, center_y - height,
                                center_x + width, center_y + height], fill='white')
            
            # Convert to base64
            buffer = io.BytesIO()
            mask_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{mask_base64}"
            
        except Exception as e:
            logger.error(f"Error creating fallback mask: {str(e)}")
            # Return minimal 1x1 pixel mask as last resort
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    def generate_inpainting_prompt(self, makeup_analysis) -> str:
        """
        Generate optimized prompt for Stable Diffusion inpainting
        
        Args:
            makeup_analysis: MakeupAnalysis object
            
        Returns:
            Optimized prompt string for inpainting
        """
        suggestions = makeup_analysis.specificSuggestions
        
        # Extract key improvement areas and create focused prompt
        prompt_parts = ["Professional makeup application"]
        
        # Analyze each suggestion and add specific improvements
        for suggestion in suggestions:
            suggestion_lower = suggestion.lower()
            
            if 'lip' in suggestion_lower:
                if 'fuller' in suggestion_lower or 'volume' in suggestion_lower:
                    prompt_parts.append("fuller, more defined lips with subtle volume")
                elif 'color' in suggestion_lower:
                    prompt_parts.append("enhanced natural lip color")
                else:
                    prompt_parts.append("improved lip definition and shape")
            
            if 'eye' in suggestion_lower:
                if 'liner' in suggestion_lower:
                    prompt_parts.append("refined eyeliner application")
                elif 'shadow' in suggestion_lower:
                    prompt_parts.append("subtle eyeshadow enhancement") 
                elif 'lift' in suggestion_lower:
                    prompt_parts.append("lifted, more defined eyes")
                else:
                    prompt_parts.append("enhanced eye definition")
            
            if 'brow' in suggestion_lower:
                prompt_parts.append("better defined, shaped eyebrows")
            
            if 'cheek' in suggestion_lower or 'blush' in suggestion_lower:
                prompt_parts.append("natural cheek color and subtle contouring")
            
            if 'skin' in suggestion_lower or 'complexion' in suggestion_lower:
                prompt_parts.append("even skin tone and smooth complexion")
            
            if 'contour' in suggestion_lower or 'sculpt' in suggestion_lower:
                prompt_parts.append("subtle facial contouring and definition")
        
        # Add quality and style descriptors
        prompt_parts.extend([
            "natural lighting",
            "high quality",
            "photorealistic", 
            "professional photography",
            "maintain facial structure and identity",
            "subtle enhancement only",
            "no dramatic changes",
            "preserve natural beauty"
        ])
        
        # Join with commas and clean up
        prompt = ", ".join(prompt_parts)
        
        # Remove duplicates and clean up
        words = prompt.split(", ")
        unique_words = []
        seen = set()
        for word in words:
            if word not in seen:
                unique_words.append(word)
                seen.add(word)
        
        final_prompt = ", ".join(unique_words)
        
        logger.info(f"Generated inpainting prompt: {final_prompt[:100]}...")
        return final_prompt