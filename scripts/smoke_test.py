#!/usr/bin/env python3
"""
Smoke Test Script for Cats vs Dogs Classification API
Validates basic functionality of the deployed service
"""
import requests
import numpy as np
from PIL import Image
import io
import json
import sys
import argparse
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SmokeTestSuite:
    """Smoke test suite for the model inference API"""
    
    def __init__(self, api_url="http://localhost:8000", timeout=30):
        self.api_url = api_url
        self.timeout = timeout
        self.session = requests.Session()
        self.results = []
    
    def _create_test_image(self, width=224, height=224):
        """Create a simple test image"""
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_health_check(self):
        """Test health check endpoint"""
        test_name = "Health Check"
        try:
            logger.info(f"Running: {test_name}")
            response = self.session.get(
                f"{self.api_url}/health",
                timeout=self.timeout
            )
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "status" in data, "Response missing 'status' field"
            assert "model_loaded" in data, "Response missing 'model_loaded' field"
            
            logger.info(f"[PASS] {test_name} passed")
            self.results.append((test_name, True, ""))
            return True
        
        except AssertionError as e:
            logger.error(f"[FAIL] {test_name} failed: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
        
        except Exception as e:
            logger.error(f"[ERROR] {test_name} error: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
    
    def test_prediction_endpoint(self):
        """Test main prediction endpoint"""
        test_name = "Prediction Endpoint"
        try:
            logger.info(f"Running: {test_name}")
            
            img_bytes = self._create_test_image()
            files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
            
            response = self.session.post(
                f"{self.api_url}/predict",
                files=files,
                timeout=self.timeout
            )
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = response.json()
            assert "prediction" in data, "Response missing 'prediction' field"
            assert "confidence" in data, "Response missing 'confidence' field"
            assert "class_probabilities" in data, "Response missing 'class_probabilities' field"
            
            # Validate prediction values
            prediction = data["prediction"]
            assert prediction in ["cat", "dog"], f"Invalid prediction: {prediction}"
            
            confidence = data["confidence"]
            assert 0 <= confidence <= 1, f"Confidence out of range: {confidence}"
            
            logger.info(f"[PASS] {test_name} passed (Prediction: {prediction}, Confidence: {confidence:.4f})")
            self.results.append((test_name, True, ""))
            return True
        
        except AssertionError as e:
            logger.error(f"[FAIL] {test_name} failed: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
        
        except Exception as e:
            logger.error(f"[ERROR] {test_name} error: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
    
    def test_base64_prediction(self):
        """Test base64 image prediction endpoint"""
        test_name = "Base64 Prediction"
        try:
            logger.info(f"Running: {test_name}")
            
            # Create and encode image
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            import base64
            img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
            
            payload = {"image_base64": img_base64}
            response = self.session.post(
                f"{self.api_url}/predict-base64",
                json=payload,
                timeout=self.timeout
            )
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = response.json()
            assert "prediction" in data, "Response missing 'prediction' field"
            assert data["prediction"] in ["cat", "dog"], f"Invalid prediction: {data['prediction']}"
            
            logger.info(f"[PASS] {test_name} passed")
            self.results.append((test_name, True, ""))
            return True
        
        except AssertionError as e:
            logger.error(f"[FAIL] {test_name} failed: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
        
        except Exception as e:
            logger.error(f"[ERROR] {test_name} error: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
    
    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint"""
        test_name = "Metrics Endpoint"
        try:
            logger.info(f"Running: {test_name}")
            
            response = self.session.get(
                f"{self.api_url}/metrics",
                timeout=self.timeout
            )
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            assert "api_requests_total" in response.text, "Metrics missing expected counter"
            assert "api_request_latency_seconds" in response.text, "Metrics missing expected histogram"
            
            logger.info(f"[PASS] {test_name} passed")
            self.results.append((test_name, True, ""))
            return True
        
        except AssertionError as e:
            logger.error(f"[FAIL] {test_name} failed: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
        
        except Exception as e:
            logger.error(f"[ERROR] {test_name} error: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
    
    def test_invalid_image(self):
        """Test handling of invalid image"""
        test_name = "Invalid Image Handling"
        try:
            logger.info(f"Running: {test_name}")
            
            files = {'file': ('test.txt', b'not an image', 'text/plain')}
            response = self.session.post(
                f"{self.api_url}/predict",
                files=files,
                timeout=self.timeout
            )
            
            # Should reject invalid image
            assert response.status_code != 200, "Should reject invalid image"
            
            logger.info(f"[PASS] {test_name} passed (correctly rejected invalid image)")
            self.results.append((test_name, True, ""))
            return True
        
        except AssertionError as e:
            logger.error(f"[FAIL] {test_name} failed: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
        
        except Exception as e:
            logger.error(f"[ERROR] {test_name} error: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
    
    def test_batch_predictions(self):
        """Test multiple predictions"""
        test_name = "Batch Predictions"
        try:
            logger.info(f"Running: {test_name}")
            
            for i in range(5):
                img_bytes = self._create_test_image()
                files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
                
                response = self.session.post(
                    f"{self.api_url}/predict",
                    files=files,
                    timeout=self.timeout
                )
                
                assert response.status_code == 200, f"Prediction {i+1} failed"
            
            logger.info(f"[PASS] {test_name} passed (5 batch predictions completed)")
            self.results.append((test_name, True, ""))
            return True
        
        except AssertionError as e:
            logger.error(f"[FAIL] {test_name} failed: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
        
        except Exception as e:
            logger.error(f"[ERROR] {test_name} error: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
    
    def run_all_tests(self):
        """Run all smoke tests"""
        logger.info("=" * 60)
        logger.info("Starting Smoke Test Suite")
        logger.info(f"Target API: {self.api_url}")
        logger.info("=" * 60)
        
        # Run tests
        self.test_health_check()
        self.test_prediction_endpoint()
        self.test_base64_prediction()
        self.test_metrics_endpoint()
        self.test_invalid_image()
        self.test_batch_predictions()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        
        passed = sum(1 for _, result, _ in self.results if result)
        total = len(self.results)
        
        for test_name, result, details in self.results:
            status = "[PASS]" if result else "[FAIL]"
            logger.info(f"{status}: {test_name} {details}")
        
        logger.info("=" * 60)
        logger.info(f"Results: {passed}/{total} tests passed")
        logger.info("=" * 60)
        
        return passed == total


def main():
    parser = argparse.ArgumentParser(description="Smoke tests for Cats vs Dogs API")
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Wait for service to be ready (seconds, default: 0)"
    )
    
    args = parser.parse_args()
    
    # Wait for service
    if args.wait > 0:
        logger.info(f"Waiting {args.wait} seconds for service to be ready...")
        time.sleep(args.wait)
    
    # Run tests
    suite = SmokeTestSuite(api_url=args.api_url, timeout=args.timeout)
    success = suite.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
