import requests
import asyncio
import aiohttp
import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ReplicateClient:
    """Client for Replicate API - Stable Diffusion Inpainting"""
    
    def __init__(self):
        """Initialize Replicate client"""
        self.base_url = "https://api.replicate.com/v1"
        self.inpainting_model = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
        self.max_poll_time = 300  # 5 minutes max
        self.poll_interval = 2    # Poll every 2 seconds
        
        logger.info("Replicate client initialized")
    
    async def generate_inpainting(self, original_image_base64: str, mask_base64: str, 
                                 prompt: str, api_key: str) -> Dict:
        """
        Generate inpainted image using Replicate API
        
        Args:
            original_image_base64: Base64 encoded original image
            mask_base64: Base64 encoded mask image
            prompt: Text prompt for inpainting
            api_key: Replicate API key
            
        Returns:
            Dictionary with success status and result
        """
        start_time = time.time()
        
        try:
            logger.info("Starting Replicate inpainting request")
            
            # Create prediction
            prediction_response = await self._create_prediction(
                original_image_base64, mask_base64, prompt, api_key
            )
            
            if not prediction_response["success"]:
                return prediction_response
            
            prediction_id = prediction_response["prediction_id"]
            logger.info(f"Created prediction: {prediction_id}")
            
            # Poll for completion
            result = await self._poll_prediction(prediction_id, api_key)
            
            processing_time = time.time() - start_time
            logger.info(f"Inpainting completed in {processing_time:.2f} seconds")
            
            if result["success"]:
                return {
                    "success": True,
                    "generated_image_url": result["generated_image_url"],
                    "processing_time": processing_time,
                    "prediction_id": prediction_id
                }
            else:
                return {
                    "success": False,
                    "error": result["error"],
                    "processing_time": processing_time,
                    "prediction_id": prediction_id
                }
                
        except Exception as e:
            logger.error(f"Error in generate_inpainting: {str(e)}")
            return {
                "success": False,
                "error": f"Replicate API error: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    async def _create_prediction(self, original_image_base64: str, mask_base64: str, 
                               prompt: str, api_key: str) -> Dict:
        """Create a new prediction on Replicate"""
        try:
            headers = {
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json"
            }
            
            # Prepare the input data
            input_data = {
                "image": original_image_base64,
                "mask": mask_base64,
                "prompt": prompt,
                "num_outputs": 1,
                "num_inference_steps": 25,  # Good balance of quality vs speed
                "guidance_scale": 7.5,      # Standard value for good prompt adherence
                "scheduler": "K_EULER_ANCESTRAL"
            }
            
            payload = {
                "version": self.inpainting_model,
                "input": input_data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/predictions",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status != 201:
                        error_text = await response.text()
                        logger.error(f"Failed to create prediction: {response.status} - {error_text}")
                        return {
                            "success": False,
                            "error": f"Failed to create prediction: {response.status} - {error_text}"
                        }
                    
                    result = await response.json()
                    return {
                        "success": True,
                        "prediction_id": result["id"],
                        "status": result["status"]
                    }
                    
        except Exception as e:
            logger.error(f"Error creating prediction: {str(e)}")
            return {
                "success": False,
                "error": f"Error creating prediction: {str(e)}"
            }
    
    async def _poll_prediction(self, prediction_id: str, api_key: str) -> Dict:
        """Poll prediction until completion"""
        try:
            headers = {
                "Authorization": f"Token {api_key}"
            }
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                while time.time() - start_time < self.max_poll_time:
                    async with session.get(
                        f"{self.base_url}/predictions/{prediction_id}",
                        headers=headers
                    ) as response:
                        
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"Failed to poll prediction: {response.status} - {error_text}")
                            return {
                                "success": False,
                                "error": f"Failed to poll prediction: {response.status} - {error_text}"
                            }
                        
                        result = await response.json()
                        status = result["status"]
                        
                        logger.debug(f"Prediction status: {status}")
                        
                        if status == "succeeded":
                            output = result.get("output")
                            if output and isinstance(output, list) and len(output) > 0:
                                return {
                                    "success": True,
                                    "generated_image_url": output[0]
                                }
                            else:
                                return {
                                    "success": False,
                                    "error": "Prediction succeeded but no output generated"
                                }
                        
                        elif status == "failed":
                            error_msg = result.get("error", "Unknown error")
                            logger.error(f"Prediction failed: {error_msg}")
                            return {
                                "success": False,
                                "error": f"Prediction failed: {error_msg}"
                            }
                        
                        elif status == "canceled":
                            return {
                                "success": False,
                                "error": "Prediction was canceled"
                            }
                        
                        elif status in ["starting", "processing"]:
                            # Still processing, wait and poll again
                            await asyncio.sleep(self.poll_interval)
                            continue
                        
                        else:
                            logger.warning(f"Unknown prediction status: {status}")
                            await asyncio.sleep(self.poll_interval)
                            continue
            
            # Timeout reached
            return {
                "success": False,
                "error": f"Prediction timed out after {self.max_poll_time} seconds"
            }
            
        except Exception as e:
            logger.error(f"Error polling prediction: {str(e)}")
            return {
                "success": False,
                "error": f"Error polling prediction: {str(e)}"
            }
    
    def sync_generate_inpainting(self, original_image_base64: str, mask_base64: str, 
                               prompt: str, api_key: str) -> Dict:
        """
        Synchronous wrapper for generate_inpainting
        Use this if you need to call from sync code
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate_inpainting(original_image_base64, mask_base64, prompt, api_key)
            )
        finally:
            loop.close()
    
    async def get_prediction_status(self, prediction_id: str, api_key: str) -> Dict:
        """Get the current status of a prediction"""
        try:
            headers = {
                "Authorization": f"Token {api_key}"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/predictions/{prediction_id}",
                    headers=headers
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        return {
                            "success": False,
                            "error": f"Failed to get prediction status: {response.status} - {error_text}"
                        }
                    
                    result = await response.json()
                    return {
                        "success": True,
                        "prediction_id": prediction_id,
                        "status": result["status"],
                        "output": result.get("output"),
                        "error": result.get("error"),
                        "created_at": result.get("created_at"),
                        "completed_at": result.get("completed_at")
                    }
                    
        except Exception as e:
            logger.error(f"Error getting prediction status: {str(e)}")
            return {
                "success": False,
                "error": f"Error getting prediction status: {str(e)}"
            }
    
    def validate_api_key(self, api_key: str) -> bool:
        """
        Validate Replicate API key format
        
        Args:
            api_key: API key to validate
            
        Returns:
            True if key format looks valid
        """
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Replicate API keys typically start with 'r8_' and are ~40 characters
        if api_key.startswith('r8_') and len(api_key) > 20:
            return True
        
        return False