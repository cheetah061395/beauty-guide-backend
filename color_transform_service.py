import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import base64
import io
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ColorTransformService:
    """Service for applying makeup through color transformations instead of inpainting"""
    
    def __init__(self):
        """Initialize color transform service"""
        # HSV values for OpenCV (H: 0-179, S: 0-255, V: 0-255)
        self.makeup_colors = {
            'lips': {
                'natural_pink': (5, 120, 200),    # Light pink/red
                'coral': (8, 150, 230),           # Coral orange-red
                'berry': (170, 180, 150),         # Berry purple-red
                'nude': (15, 80, 200)             # Nude pink-brown
            },
            'eyeshadow': {
                'natural_brown': (15, 100, 130),  # Natural brown
                'bronze': (12, 150, 155),         # Bronze brown
                'neutral': (10, 50, 100),         # Neutral taupe
                'warm_taupe': (18, 80, 130)       # Warm taupe
            },
            'blush': {
                'natural_pink': (3, 100, 200),    # Natural pink
                'peach': (10, 130, 230),          # Peach
                'rose': (2, 120, 180),            # Rose pink
                'coral': (8, 150, 200)            # Coral pink
            }
        }
        logger.info("Color transform service initialized")
    
    def _parse_ai_recommendations(self, makeup_analysis) -> Dict:
        """Parse AI analysis to determine specific colors and intensities"""
        recommendations = {
            'lips': {'apply': False, 'color': None, 'intensity': 0.0},
            'eyes': {'apply': False, 'color': None, 'intensity': 0.0},
            'cheeks': {'apply': False, 'color': None, 'intensity': 0.0},
            'eyebrows': {'apply': False, 'color': None, 'intensity': 0.0}
        }
        
        # Combine all AI text for analysis
        all_text = " ".join([
            makeup_analysis.generalFeedback,
            *makeup_analysis.specificSuggestions
        ]).lower()
        
        # Parse lip recommendations
        if any(word in all_text for word in ['lip', 'mouth', 'lipstick']):
            recommendations['lips']['apply'] = True
            
            # Detect specific colors
            if any(word in all_text for word in ['pink', 'rose']):
                recommendations['lips']['color'] = 'natural_pink'
            elif any(word in all_text for word in ['coral', 'orange']):
                recommendations['lips']['color'] = 'coral'
            elif any(word in all_text for word in ['berry', 'deep', 'red']):
                recommendations['lips']['color'] = 'berry'
            elif any(word in all_text for word in ['nude', 'natural']):
                recommendations['lips']['color'] = 'nude'
            else:
                recommendations['lips']['color'] = 'natural_pink'  # default
            
            # Detect intensity
            recommendations['lips']['intensity'] = self._parse_intensity(all_text, 'lip')
        
        # Parse eye recommendations
        if any(word in all_text for word in ['eye', 'shadow', 'lid']):
            recommendations['eyes']['apply'] = True
            recommendations['eyes']['color'] = 'natural_brown'
            recommendations['eyes']['intensity'] = self._parse_intensity(all_text, 'eye')
        
        # Parse cheek recommendations
        if any(word in all_text for word in ['cheek', 'blush', 'flush']):
            recommendations['cheeks']['apply'] = True
            
            if any(word in all_text for word in ['peach', 'warm']):
                recommendations['cheeks']['color'] = 'peach'
            elif any(word in all_text for word in ['rose', 'pink']):
                recommendations['cheeks']['color'] = 'rose'
            else:
                recommendations['cheeks']['color'] = 'natural_pink'
                
            recommendations['cheeks']['intensity'] = self._parse_intensity(all_text, 'cheek')
        
        # Parse eyebrow recommendations
        if any(word in all_text for word in ['brow', 'eyebrow']):
            recommendations['eyebrows']['apply'] = True
            recommendations['eyebrows']['color'] = 'natural_brown'
            recommendations['eyebrows']['intensity'] = self._parse_intensity(all_text, 'brow')
        
        logger.info(f"Parsed AI recommendations: {recommendations}")
        return recommendations
    
    def _parse_intensity(self, text: str, feature: str) -> float:
        """Parse intensity keywords for specific feature"""
        feature_text = text
        
        # Look for intensity keywords
        if any(word in feature_text for word in ['very subtle', 'barely', 'hint']):
            return 0.15
        elif any(word in feature_text for word in ['subtle', 'light', 'gentle', 'soft']):
            return 0.25
        elif any(word in feature_text for word in ['natural', 'enhance']):
            return 0.35
        elif any(word in feature_text for word in ['moderate', 'define', 'improve']):
            return 0.45
        elif any(word in feature_text for word in ['bold', 'strong', 'dramatic']):
            return 0.6
        elif any(word in feature_text for word in ['vibrant', 'intense', 'pronounced']):
            return 0.75
        else:
            return 0.3  # Default moderate intensity
    
    def apply_makeup_color_transform(self, image_data: bytes, landmarks: List[Dict], 
                                   makeup_analysis, target_regions: List[Dict],
                                   makeup_intensity: float = 0.6) -> Dict:
        """
        Apply makeup using color transformations instead of inpainting
        
        Args:
            image_data: Raw image bytes
            landmarks: MediaPipe facial landmarks
            makeup_analysis: MakeupAnalysis object
            target_regions: List of target regions to enhance
            makeup_intensity: Intensity of makeup application (0.0-1.0)
            
        Returns:
            Dictionary with success status and result
        """
        try:
            logger.info("Starting AI-driven color transform makeup application")
            
            # Parse AI recommendations first
            ai_recommendations = self._parse_ai_recommendations(makeup_analysis)
            
            # Convert image to PIL and numpy
            image = Image.open(io.BytesIO(image_data))
            image_np = np.array(image)
            image_width, image_height = image.size
            
            # Convert to HSV for better color manipulation
            hsv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Process each target region based on AI recommendations
            regions_processed = []
            for region in target_regions:
                region_name = region['name']
                logger.info(f"Applying color transform to: {region_name}")
                
                # Get region landmarks and create mask
                region_mask = self._create_region_mask(
                    landmarks, region_name, image_width, image_height
                )
                
                if region_mask is None:
                    logger.warning(f"Could not create mask for region: {region_name}")
                    continue
                
                # Apply makeup based on AI recommendations
                if 'lip' in region_name and ai_recommendations['lips']['apply']:
                    self._apply_ai_driven_enhancement(
                        hsv_image, region_mask, 'lips', ai_recommendations['lips']
                    )
                    regions_processed.append('lips')
                elif 'eye' in region_name and 'brow' not in region_name and ai_recommendations['eyes']['apply']:
                    self._apply_ai_driven_enhancement(
                        hsv_image, region_mask, 'eyes', ai_recommendations['eyes']
                    )
                    regions_processed.append('eyes')
                elif 'cheek' in region_name and ai_recommendations['cheeks']['apply']:
                    self._apply_ai_driven_enhancement(
                        hsv_image, region_mask, 'cheeks', ai_recommendations['cheeks']
                    )
                    regions_processed.append('cheeks')
                elif 'brow' in region_name and ai_recommendations['eyebrows']['apply']:
                    self._apply_ai_driven_enhancement(
                        hsv_image, region_mask, 'eyebrows', ai_recommendations['eyebrows']
                    )
                    regions_processed.append('eyebrows')
                else:
                    logger.info(f"Skipping {region_name} - not recommended by AI")
            
            # Convert back to RGB
            enhanced_image_np = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)
            enhanced_image = Image.fromarray(enhanced_image_np)
            
            # Apply subtle smoothing for natural finish
            enhanced_image = self._apply_natural_smoothing(enhanced_image)
            
            # Convert to base64 for return
            buffer = io.BytesIO()
            enhanced_image.save(buffer, format='JPEG', quality=95)
            buffer.seek(0)
            
            enhanced_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            enhanced_data_url = f"data:image/jpeg;base64,{enhanced_base64}"
            
            logger.info("Color transform makeup application completed successfully")
            
            return {
                "success": True,
                "generated_image_url": enhanced_data_url,
                "method": "ai_driven_color_transform",
                "regions_processed": list(set(regions_processed)),
                "ai_recommendations_applied": ai_recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in color transform makeup application: {str(e)}")
            return {
                "success": False,
                "error": f"Color transform error: {str(e)}"
            }
    
    def _create_region_mask(self, landmarks: List[Dict], region_name: str, 
                          image_width: int, image_height: int) -> Optional[np.ndarray]:
        """Create a smooth mask for the specified region"""
        try:
            # Use the same landmark indices from mask_generator
            region_indices = self._get_region_landmark_indices(region_name)
            
            if not region_indices:
                return None
            
            # Extract points
            points = []
            for idx in region_indices:
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    points.append((landmark['pixel_x'], landmark['pixel_y']))
            
            if len(points) < 3:
                return None
            
            # Create mask
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            points_np = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points_np], 255)
            
            # Apply Gaussian blur for smooth edges
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error creating region mask for {region_name}: {str(e)}")
            return None
    
    def _apply_ai_driven_enhancement(self, hsv_image: np.ndarray, mask: np.ndarray, 
                                   feature_type: str, ai_recommendation: Dict):
        """Apply enhancement based on AI recommendations"""
        try:
            color_name = ai_recommendation['color']
            intensity = ai_recommendation['intensity']
            
            logger.info(f"Applying {feature_type} enhancement: {color_name} at intensity {intensity}")
            
            # Get the appropriate color
            if feature_type == 'lips' and color_name in self.makeup_colors['lips']:
                target_color = self.makeup_colors['lips'][color_name]
                self._apply_color_to_region(hsv_image, mask, target_color, intensity)
                
            elif feature_type == 'eyes' and color_name in self.makeup_colors['eyeshadow']:
                target_color = self.makeup_colors['eyeshadow'][color_name]
                self._apply_color_to_region(hsv_image, mask, target_color, intensity)
                
            elif feature_type == 'cheeks' and color_name in self.makeup_colors['blush']:
                target_color = self.makeup_colors['blush'][color_name]
                self._apply_color_to_region(hsv_image, mask, target_color, intensity)
                
            elif feature_type == 'eyebrows':
                # For eyebrows, just darken slightly based on intensity
                mask_norm = mask.astype(np.float32) / 255.0
                darkening_factor = intensity * 0.4  # Reduced for natural look
                hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * (1 - darkening_factor * mask_norm), 0, 255)
            
            logger.debug(f"Applied {feature_type} enhancement successfully")
            
        except Exception as e:
            logger.error(f"Error applying {feature_type} enhancement: {str(e)}")
    
    def _apply_lip_enhancement(self, hsv_image: np.ndarray, mask: np.ndarray, 
                             makeup_analysis, intensity: float):
        """Apply natural lip color enhancement"""
        try:
            # Determine lip color based on analysis
            suggestions_text = " ".join(makeup_analysis.specificSuggestions).lower()
            
            if any(word in suggestions_text for word in ['coral', 'orange']):
                target_color = self.makeup_colors['lips']['coral']
            elif any(word in suggestions_text for word in ['berry', 'deep', 'dark']):
                target_color = self.makeup_colors['lips']['berry']
            elif any(word in suggestions_text for word in ['nude', 'natural', 'subtle']):
                target_color = self.makeup_colors['lips']['nude']
            else:
                target_color = self.makeup_colors['lips']['natural_pink']
            
            # Apply color enhancement with reduced intensity for natural look
            self._apply_color_to_region(hsv_image, mask, target_color, intensity * 0.4)
            
            # Add subtle saturation boost
            mask_norm = mask.astype(np.float32) / 255.0
            saturation_boost = intensity * 0.2
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * (1 + saturation_boost * mask_norm), 0, 255)
            
            logger.debug("Applied lip enhancement")
            
        except Exception as e:
            logger.error(f"Error applying lip enhancement: {str(e)}")
    
    def _apply_eyeshadow_enhancement(self, hsv_image: np.ndarray, mask: np.ndarray,
                                   makeup_analysis, intensity: float):
        """Apply subtle eyeshadow color"""
        try:
            # Use neutral brown tones for natural look
            target_color = self.makeup_colors['eyeshadow']['natural_brown']
            
            # Apply with lower intensity for subtlety
            self._apply_color_to_region(hsv_image, mask, target_color, intensity * 0.2)
            
            logger.debug("Applied eyeshadow enhancement")
            
        except Exception as e:
            logger.error(f"Error applying eyeshadow enhancement: {str(e)}")
    
    def _apply_blush_enhancement(self, hsv_image: np.ndarray, mask: np.ndarray,
                               makeup_analysis, intensity: float):
        """Apply natural blush color"""
        try:
            # Determine blush color
            suggestions_text = " ".join(makeup_analysis.specificSuggestions).lower()
            
            if any(word in suggestions_text for word in ['peach', 'warm']):
                target_color = self.makeup_colors['blush']['peach']
            elif any(word in suggestions_text for word in ['rose', 'pink']):
                target_color = self.makeup_colors['blush']['rose']
            else:
                target_color = self.makeup_colors['blush']['natural_pink']
            
            # Apply with gentle intensity
            self._apply_color_to_region(hsv_image, mask, target_color, intensity * 0.3)
            
            logger.debug("Applied blush enhancement")
            
        except Exception as e:
            logger.error(f"Error applying blush enhancement: {str(e)}")
    
    def _apply_eyebrow_enhancement(self, hsv_image: np.ndarray, mask: np.ndarray,
                                 makeup_analysis, intensity: float):
        """Enhance eyebrow definition without changing structure"""
        try:
            # Darken existing eyebrow hair
            mask_norm = mask.astype(np.float32) / 255.0
            
            # Reduce brightness (darken)
            darkening_factor = intensity * 0.3
            hsv_image[:, :, 2] = hsv_image[:, :, 2] * (1 - darkening_factor * mask_norm)
            
            # Increase saturation slightly
            hsv_image[:, :, 1] = hsv_image[:, :, 1] * (1 + 0.2 * mask_norm)
            
            logger.debug("Applied eyebrow enhancement")
            
        except Exception as e:
            logger.error(f"Error applying eyebrow enhancement: {str(e)}")
    
    def _apply_color_to_region(self, hsv_image: np.ndarray, mask: np.ndarray, 
                             target_hsv: Tuple[int, int, int], intensity: float):
        """Apply specific color to masked region with blending"""
        try:
            mask_norm = mask.astype(np.float32) / 255.0
            target_h, target_s, target_v = target_hsv
            
            # Blend hue (OpenCV H range: 0-179)
            current_h = hsv_image[:, :, 0]
            blended_h = current_h * (1 - intensity * mask_norm) + target_h * intensity * mask_norm
            hsv_image[:, :, 0] = np.clip(blended_h, 0, 179)
            
            # Blend saturation (OpenCV S range: 0-255)
            current_s = hsv_image[:, :, 1]
            blended_s = current_s * (1 - intensity * mask_norm) + target_s * intensity * mask_norm
            hsv_image[:, :, 1] = np.clip(blended_s, 0, 255)
            
            # Slight value adjustment (OpenCV V range: 0-255)
            current_v = hsv_image[:, :, 2]
            blended_v = current_v * (1 - intensity * mask_norm * 0.3) + target_v * intensity * mask_norm * 0.3
            hsv_image[:, :, 2] = np.clip(blended_v, 0, 255)
            
        except Exception as e:
            logger.error(f"Error applying color to region: {str(e)}")
    
    def _apply_natural_smoothing(self, image: Image.Image) -> Image.Image:
        """Apply subtle smoothing for natural makeup finish"""
        try:
            # Light smoothing only in makeup areas
            image_np = np.array(image)
            
            # Apply very subtle bilateral filter for skin smoothing
            smoothed = cv2.bilateralFilter(image_np, 5, 10, 10)
            
            # Blend original and smoothed (keep mostly original)
            blend_factor = 0.15  # Very subtle
            result = image_np * (1 - blend_factor) + smoothed * blend_factor
            
            return Image.fromarray(result.astype(np.uint8))
            
        except Exception as e:
            logger.error(f"Error applying natural smoothing: {str(e)}")
            return image
    
    def _get_region_landmark_indices(self, region_name: str) -> List[int]:
        """Get landmark indices for facial regions (same as mask_generator)"""
        region_indices = {
            'lips': [
                61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82
            ],
            'left_eye': [
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28, 27, 29, 30
            ],
            'right_eye': [
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398,
                359, 255, 339, 254, 253, 252, 256, 341, 463, 414, 286, 258, 257, 259, 260
            ],
            'left_eyebrow': [46, 53, 52, 51, 48, 115, 131, 134, 102, 49, 220, 305, 70, 63, 105, 66, 107, 55, 65],
            'right_eyebrow': [276, 283, 282, 281, 278, 344, 360, 363, 331, 279, 440, 75, 300, 293, 334, 296, 336, 285, 295],
            'left_cheek': [116, 117, 118, 119, 120, 121, 126, 142, 36, 205, 206, 207, 213, 192, 147, 123, 137, 177, 50, 36],
            'right_cheek': [345, 346, 347, 348, 349, 350, 355, 371, 266, 425, 426, 427, 436, 416, 376, 352, 366, 401, 280, 266]
        }
        
        return region_indices.get(region_name, [])
    
    def calculate_makeup_intensity(self, makeup_analysis) -> float:
        """Calculate appropriate makeup intensity based on analysis"""
        all_text = " ".join(makeup_analysis.specificSuggestions).lower()
        
        # Keywords that suggest different intensities
        subtle_keywords = ['subtle', 'light', 'natural', 'slight', 'gentle', 'soft']
        moderate_keywords = ['enhance', 'define', 'improve', 'better', 'moderate']
        strong_keywords = ['dramatic', 'bold', 'strong', 'pronounced', 'vibrant']
        
        if any(keyword in all_text for keyword in strong_keywords):
            return 0.8  # Strong makeup
        elif any(keyword in all_text for keyword in subtle_keywords):
            return 0.4  # Subtle makeup
        else:
            return 0.6  # Moderate makeup (default)