#!/usr/bin/env python
"""
Smoke Tests - API Health Check and Prediction Validation
Post-deploy smoke tests for the Cat vs Dog classification API
"""

import sys
import time
import argparse
import requests
import numpy as np
from PIL import Image
import io
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

class APISmokeTests:
    """API Smoke tests for health check and predictions"""
    
    def __init__(self, api_url="http://localhost:8000", timeout=10):
        self.api_url = api_url
        self.timeout = timeout
        self.session = requests.Session()
        self.results = []
    
    def _create_test_image(self, width=224, height=224):
        """Create a test image for prediction"""
        img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        return img_bytes
    
    def test_health_check(self):
        """Test /health endpoint"""
        test_name = "Health Check"
        try:
            logger.info(f"Testing: {test_name}")
            response = self.session.get(
                f"{self.api_url}/health",
                timeout=self.timeout
            )
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert "status" in data, "Response missing 'status' field"
            
            logger.info(f"  [OK] API is healthy")
            self.results.append((test_name, True, ""))
            return True
        
        except requests.ConnectionError as e:
            logger.error(f"  [FAIL] Cannot connect to API at {self.api_url}")
            self.results.append((test_name, False, f"Connection error: {self.api_url}"))
            return False
        
        except AssertionError as e:
            logger.error(f"  [FAIL] {test_name} assertion failed: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
        
        except Exception as e:
            logger.error(f"  [FAIL] {test_name} error: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
    
    def test_prediction_endpoint(self):
        """Test /predict endpoint with sample image"""
        test_name = "Prediction Endpoint"
        try:
            logger.info(f"Testing: {test_name}")
            
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
            
            prediction = data["prediction"]
            confidence = data["confidence"]
            
            assert prediction in ["cat", "dog"], f"Invalid prediction: {prediction}"
            assert 0 <= confidence <= 1, f"Confidence out of range: {confidence}"
            
            logger.info(f"  [OK] Prediction successful: {prediction} ({confidence:.2%})")
            self.results.append((test_name, True, ""))
            return True
        
        except AssertionError as e:
            logger.error(f"  [FAIL] {test_name} assertion failed: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
        
        except Exception as e:
            logger.error(f"  [FAIL] {test_name} error: {str(e)}")
            self.results.append((test_name, False, str(e)))
            return False
    
    def run_all_tests(self):
        """Run all smoke tests"""
        print("\n" + "=" * 70)
        print("  Cat vs Dog Classification API - Smoke Tests")
        print("=" * 70 + "\n")
        
        # Run tests
        self.test_health_check()
        self.test_prediction_endpoint()
        
        # Print summary
        print("\n" + "=" * 70)
        print("  Test Summary")
        print("=" * 70)
        
        passed = sum(1 for _, result, _ in self.results if result)
        total = len(self.results)
        
        for test_name, result, details in self.results:
            status = "[PASS]" if result else "[FAIL]"
            if details:
                logger.info(f"{status}: {test_name} ({details})")
            else:
                logger.info(f"{status}: {test_name}")
        
        print("\n" + "-" * 70)
        logger.info(f"Results: {passed}/{total} tests passed")
        print("=" * 70 + "\n")
        
        if passed == total:
            logger.info("[OK] All smoke tests passed! API is ready for use.")
        else:
            logger.warning(f"[FAIL] {total - passed} test(s) failed. Check configuration.")
        
        return passed == total


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Smoke tests for Cat vs Dog classification API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/smoke_test.py                               # Default (localhost:8000)
  python scripts/smoke_test.py --api-url http://api:8000    # Custom API URL
  python scripts/smoke_test.py --wait 10                     # Wait 10s for API to start
        """
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="API URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=0,
        help="Wait for API to be ready (seconds, default: 0)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Request timeout in seconds (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Wait for API if requested
    if args.wait > 0:
        logger.info(f"Waiting {args.wait}s for API to be ready...")
        time.sleep(args.wait)
    
    # Run smoke tests
    suite = APISmokeTests(api_url=args.api_url, timeout=args.timeout)
    success = suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
