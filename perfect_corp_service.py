import requests
import base64
import time
import logging
from typing import Dict, Optional, List
import json
from PIL import Image
import io
import os
import uuid

logger = logging.getLogger(__name__)

# 🚀 VERIFICATION: NEW CODE VERSION 3.0 LOADING 
print("🚀🚀🚀 NEW PERFECT CORP CODE VERSION 3.0 LOADED! 🚀🚀🚀")

class PerfectCorpService:
    """Service for integrating with Perfect Corp YouCam AI API"""
    
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        # Use the correct API server from documentation
        self.base_url = "https://yce-api-01.makeupar.com"
        # Bearer token authentication as per documentation
        self.auth_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def save_image_locally(self, image_data: bytes, filename: str) -> str:
        """Save image locally and return public URL"""
        import uuid
        import os
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate unique filename
        file_extension = os.path.splitext(filename)[1] or '.jpg'
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(uploads_dir, unique_filename)
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        # Return public URL using environment variable or localhost fallback
        base_url = os.getenv("BASE_URL", "http://localhost:8000")
        public_url = f"{base_url}/uploads/{unique_filename}"
        logger.info(f"Saved image locally: {file_path}, public URL: {public_url}")
        return public_url
    
    def _create_makeup_vto_task(self, file_url: str, makeup_effects: List[Dict]) -> str:
        """Step 2: Create makeup VTO task using documented task endpoint"""
        try:
            # Use documented makeup VTO task endpoint
            url = f"{self.base_url}/s2s/v2.0/task/makeup-vto"
            
            data = {
                "src_file_url": file_url,
                "effects": makeup_effects,
                "version": "1.0"
            }
            
            logger.info(f"Creating makeup VTO task at: {url}")
            logger.info(f"Makeup effects: {makeup_effects}")
            response = requests.post(url, headers=self.auth_headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                task_id = (result.get("taskId") or 
                          result.get("task_id") or 
                          result.get("data", {}).get("taskId") or 
                          result.get("data", {}).get("task_id"))
                
                if task_id:
                    logger.info(f"Makeup VTO task created, task ID: {task_id}")
                    return task_id
                else:
                    raise Exception(f"No task_id in VTO response: {result}")
            else:
                error_text = response.text
                logger.error(f"Makeup VTO task creation failed with {response.status_code}: {error_text}")
                raise Exception(f"Makeup VTO task failed: {response.status_code} - {error_text}")
                
        except Exception as e:
            logger.error(f"Error creating makeup VTO task: {str(e)}")
            raise
    
    def _poll_task_status(self, task_id: str, max_attempts: int = 30) -> Dict:
        """Step 3: Poll for AI task completion and return result"""
        try:
            # Use documented makeup VTO status endpoint
            url = f"{self.base_url}/s2s/v2.0/task/makeup-vto/{task_id}"
            
            for attempt in range(max_attempts):
                response = requests.get(url, headers=self.auth_headers, timeout=10)
                response.raise_for_status()
                
                result = response.json()
                logger.debug(f"Polling response: {result}")
                status = result.get("status") or result.get("data", {}).get("status")
                logger.debug(f"Extracted status: {status}")
                
                if status == "completed":
                    logger.info(f"AI makeup VTO task completed: {task_id}")
                    return result
                elif status == "failed":
                    error = result.get("error") or result.get("data", {}).get("error", "Unknown error")
                    raise Exception(f"AI makeup VTO task failed: {error}")
                elif status in ["pending", "processing", "uploaded", "running"]:
                    logger.debug(f"Task {task_id} still processing, attempt {attempt + 1}/{max_attempts}")
                    time.sleep(3)  # Wait 3 seconds between polls
                else:
                    logger.warning(f"Unknown task status: {status}")
                    time.sleep(3)
            
            raise Exception("AI makeup VTO task timeout - maximum attempts reached")
            
        except Exception as e:
            logger.error(f"Error polling Face AI task status: {str(e)}")
            raise
    
    
    def _convert_to_perfect_corp_effects(self, perfect_corp_effects) -> List[Dict]:
        """Convert structured AI effects to Perfect Corp API format"""
        logger.info("🔥 NEW CODE LOADED - Using _convert_to_perfect_corp_effects function!")
        effects = []
        
        if not perfect_corp_effects:
            logger.warning("No perfectCorpEffects data provided by AI")
            return []
        
        logger.info(f"Processing {len(perfect_corp_effects)} effects from AI: {perfect_corp_effects}")
        
        for effect in perfect_corp_effects:
            try:
                category = effect.category
                color = effect.color
                texture = effect.texture
                intensity = effect.intensity
                
                # Validate required fields
                if not all([category, color, texture, intensity]):
                    logger.warning(f"❌ Skipping incomplete effect: {effect}")
                    continue
                
                # Convert to Perfect Corp format based on category
                if category == "blush":
                    perfect_corp_effect = {
                        "category": "blush",
                        "pattern": {"name": "Circle1"},  # Using verified pattern from catalog
                        "palettes": [
                            {
                                "color": color,
                                "texture": texture,
                                "colorIntensity": intensity
                            }
                        ]
                    }
                    effects.append(perfect_corp_effect)
                    logger.info(f"✅ Added blush effect: {color} {texture} intensity {intensity}")
                
                elif category == "lip_color":
                    perfect_corp_effect = {
                        "category": "lip_color",
                        "shape": {"name": "original"},  # Using verified shape from catalog
                        "palettes": [
                            {
                                "color": color,
                                "texture": texture
                            }
                        ]
                    }
                    effects.append(perfect_corp_effect)
                    logger.info(f"✅ Added lip color effect: {color} {texture}")
                
                elif category == "eye_liner":
                    perfect_corp_effect = {
                        "category": "eye_liner",
                        "pattern": {"name": "FullRim1"},  # Using verified pattern from catalog
                        "palettes": [
                            {
                                "color": color,
                                "texture": texture,
                                "colorIntensity": intensity
                            }
                        ]
                    }
                    effects.append(perfect_corp_effect)
                    logger.info(f"✅ Added eye liner effect: {color} {texture} intensity {intensity}")
                
                elif category == "eyeshadow":
                    perfect_corp_effect = {
                        "category": "eyeshadow",
                        "pattern": {"name": "Basic1"},  # Using basic eyeshadow pattern
                        "palettes": [
                            {
                                "color": color,
                                "texture": texture,
                                "colorIntensity": intensity
                            }
                        ]
                    }
                    effects.append(perfect_corp_effect)
                    logger.info(f"✅ Added eyeshadow effect: {color} {texture} intensity {intensity}")
                
                else:
                    logger.warning(f"❌ Unsupported category: {category}")
            
            except Exception as e:
                logger.warning(f"❌ Skipping effect due to error: {e}")
        
        # Fallback: if no effects from AI, add default blush
        if not effects:
            logger.info("No valid effects from AI, adding default blush fallback")
            effects = [{
                "category": "blush",
                "pattern": {"name": "Circle1"},
                "palettes": [
                    {"color": "#FFCBA4", "texture": "matte", "colorIntensity": 40}
                ]
            }]
        
        logger.info(f"Generated {len(effects)} Perfect Corp makeup effects")
        return effects
    
    
    
    
    def apply_ai_driven_makeup(self, image_data: bytes, makeup_analysis) -> Dict:
        """
        Complete 3-step workflow: upload image -> create VTO task -> poll for result
        
        Args:
            image_data: Raw image bytes
            makeup_analysis: MakeupAnalysis object with AI recommendations
            
        Returns:
            Dictionary with success status and result
        """
        try:
            logger.info("Starting Perfect Corp AI Makeup VTO workflow")
            
            # Step 1: Convert structured AI effects to Perfect Corp format
            logger.info(f"AI provided perfectCorpEffects: {makeup_analysis.perfectCorpEffects}")
            makeup_effects = self._convert_to_perfect_corp_effects(makeup_analysis.perfectCorpEffects)
            
            # Step 2: Save image locally and get public URL
            file_url = self.save_image_locally(image_data, "makeup_photo.jpg")
            
            # Step 3: Create makeup VTO task (with fallback for partial effects)
            task_id = None
            working_effects = makeup_effects.copy()
            
            # Try with all effects first
            try:
                task_id = self._create_makeup_vto_task(file_url, working_effects)
                logger.info(f"✅ All effects accepted: {[e['category'] for e in working_effects]}")
            except Exception as e:
                logger.warning(f"❌ All effects failed: {e}")
                
                # Try with just blush (most likely to work)
                try:
                    blush_only = [e for e in working_effects if e['category'] == 'blush']
                    if blush_only:
                        task_id = self._create_makeup_vto_task(file_url, blush_only)
                        working_effects = blush_only
                        logger.info(f"✅ Fallback to blush only successful")
                    else:
                        raise Exception("No blush effect available for fallback")
                except Exception as e2:
                    logger.error(f"❌ Even blush fallback failed: {e2}")
                    raise e  # Re-raise original error
            
            # Step 4: Poll for task completion
            task_result = self._poll_task_status(task_id)
            
            # Step 5: Extract result image URL from response
            result_image_url = (task_result.get("resultImageUrl") or 
                              task_result.get("data", {}).get("resultImageUrl") or
                              task_result.get("resultUrl"))
            
            if not result_image_url:
                raise Exception(f"No result image URL in Perfect Corp response: {task_result}")
            
            logger.info("Perfect Corp AI Makeup VTO completed successfully")
            
            return {
                "success": True,
                "generated_image_url": result_image_url,
                "method": "perfect_corp_makeup_vto",
                "makeup_effects": working_effects,
                "effects_applied": [e['category'] for e in working_effects],
                "task_id": task_id,
                "file_url": file_url
            }
            
        except Exception as e:
            logger.error(f"Error in Perfect Corp AI Makeup VTO: {str(e)}")
            return {
                "success": False,
                "error": f"Perfect Corp VTO error: {str(e)}"
            }
    
    def health_check(self) -> Dict:
        """Check if Perfect Corp AI Makeup VTO API is accessible"""
        try:
            # Test API access with documented endpoint
            url = f"{self.base_url}/s2s/v2.0/task/ai-task"
            response = requests.head(url, headers=self.auth_headers, timeout=10)
            
            # Success if not auth failure
            if response.status_code not in [401, 403]:
                return {
                    "success": True,
                    "message": "Perfect Corp AI Makeup VTO API is accessible",
                    "api_key_configured": bool(self.api_key),
                    "base_url": self.base_url,
                    "auth_method": "Bearer Token",
                    "api_version": "v2.0"
                }
            else:
                return {
                    "success": False,
                    "message": "API authentication failed",
                    "api_key_configured": bool(self.api_key),
                    "base_url": self.base_url,
                    "auth_method": "Bearer Token (failed)"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Perfect Corp AI Makeup VTO API health check failed: {str(e)}"
            }