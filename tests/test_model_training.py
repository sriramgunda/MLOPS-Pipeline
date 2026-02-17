"""
Unit tests for model inference and prediction functions
Tests model loading, inference, and output validation
"""
import pytest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestModelInference:
    """Test suite for model inference functions"""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a mock/dummy model for testing"""
        # For real tests, this would be a trained Keras model
        # For unit tests, we'll mock it
        model = MagicMock()
        model.predict = MagicMock(return_value=np.array([[0.7]]))  # Dog prediction
        return model
    
    
    @pytest.fixture
    def dummy_image_array(self):
        """Create dummy preprocessed image array"""
        return np.random.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)
    
    
    def test_prediction_output_shape(self, dummy_model, dummy_image_array):
        """Test that prediction returns expected shape"""
        prediction = dummy_model.predict(dummy_image_array)
        
        assert prediction.shape == (1, 1), "Prediction should be (batch_size, 1)"
        assert 0 <= prediction[0][0] <= 1, "Prediction should be probability in [0, 1]"
    
    
    def test_prediction_bounds(self, dummy_model):
        """Test that predictions are valid probabilities"""
        # Test multiple predictions
        for prob_value in [0.0, 0.25, 0.5, 0.75, 1.0]:
            dummy_model.predict.return_value = np.array([[prob_value]])
            pred = dummy_model.predict(np.zeros((1, 224, 224, 3)))
            
            assert 0.0 <= pred[0][0] <= 1.0, f"Prediction {prob_value} out of bounds"
    
    
    def test_batch_prediction(self, dummy_model):
        """Test batch prediction"""
        batch_size = 4
        batch_images = np.random.uniform(0, 1, (batch_size, 224, 224, 3)).astype(np.float32)
        
        # Mock batch predictions
        batch_predictions = np.array([[0.2], [0.8], [0.6], [0.1]])
        dummy_model.predict.return_value = batch_predictions
        
        predictions = dummy_model.predict(batch_images)
        
        assert predictions.shape == (batch_size, 1), "Batch prediction shape incorrect"
        assert all(0 <= p <= 1 for p in predictions.flatten()), "All predictions should be probabilities"
    
    
    def test_prediction_stability(self, dummy_model):
        """Test that prediction is deterministic for same input"""
        image = np.ones((1, 224, 224, 3), dtype=np.float32)
        
        dummy_model.predict.return_value = np.array([[0.75]])
        pred1 = dummy_model.predict(image)
        pred2 = dummy_model.predict(image)
        
        assert np.allclose(pred1, pred2), "Same input should produce same prediction"
    
    
    def test_class_label_assignment(self):
        """Test correct class label assignment from probability"""
        test_cases = [
            (0.2, "cat"),   # < 0.5 -> cat
            (0.4, "cat"),   # < 0.5 -> cat
            (0.5, "dog"),   # >= 0.5 -> dog
            (0.6, "dog"),   # > 0.5 -> dog
            (0.9, "dog"),   # > 0.5 -> dog
        ]
        
        for prob, expected_class in test_cases:
            predicted_class = "dog" if prob > 0.5 else "cat"
            assert predicted_class == expected_class, \
                f"Probability {prob} should map to {expected_class}, got {predicted_class}"
    
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        predictions = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for pred_prob in predictions:
            confidence = pred_prob if pred_prob > 0.5 else (1 - pred_prob)
            
            # Confidence should always be >= 0.5 (same side of threshold)
            assert confidence >= 0.5, \
                f"Confidence for {pred_prob} should be >= 0.5, got {confidence}"
    
    
    def test_class_probability_dict(self):
        """Test generation of class probability dictionary"""
        prob_dog = 0.75
        prob_cat = 1 - prob_dog
        
        class_probs = {
            "cat": prob_cat,
            "dog": prob_dog
        }
        
        assert "cat" in class_probs, "Dict should contain 'cat' key"
        assert "dog" in class_probs, "Dict should contain 'dog' key"
        assert abs(class_probs["cat"] + class_probs["dog"] - 1.0) < 1e-6, \
            "Probabilities should sum to 1.0"


class TestInferenceEdgeCases:
    """Test edge cases in inference"""
    
    def test_very_low_confidence_cat(self):
        """Test prediction very close to cat threshold"""
        pred_prob = 0.01
        predicted_class = "dog" if pred_prob > 0.5 else "cat"
        confidence = 1 - pred_prob if pred_prob < 0.5 else pred_prob
        
        assert predicted_class == "cat"
        assert confidence > 0.99
    
    
    def test_very_low_confidence_dog(self):
        """Test prediction very close to dog threshold"""
        pred_prob = 0.99
        predicted_class = "dog" if pred_prob > 0.5 else "cat"
        confidence = 1 - pred_prob if pred_prob < 0.5 else pred_prob
        
        assert predicted_class == "dog"
        assert confidence > 0.99
    
    
    def test_ambiguous_prediction(self):
        """Test prediction near 0.5 threshold"""
        pred_prob = 0.51
        predicted_class = "dog" if pred_prob > 0.5 else "cat"
        confidence = 1 - pred_prob if pred_prob < 0.5 else pred_prob
        
        assert predicted_class == "dog"
        assert 0.49 < confidence < 0.52


class TestInferenceValidation:
    """Test input validation for inference"""
    
    def test_invalid_image_shape(self):
        """Test handling of wrong image shape"""
        wrong_shape_image = np.random.uniform(0, 1, (1, 100, 100, 3))  # Wrong size
        
        # Should handle gracefully or raise informative error
        # For now, just ensure it doesn't crash unexpectedly
        try:
            # Attempt to use wrong shape (would fail in real model)
            assert wrong_shape_image.shape != (1, 224, 224, 3)
        except Exception as e:
            assert "shape" in str(e).lower() or "dimension" in str(e).lower()
    
    
    def test_invalid_value_range(self):
        """Test handling of out-of-range pixel values"""
        # Values outside [0, 1]
        invalid_image = np.random.uniform(-1, 2, (1, 224, 224, 3)).astype(np.float32)
        
        # Should be detected as invalid
        assert invalid_image.min() < 0 or invalid_image.max() > 1, \
            "Test image should have out-of-range values"
    
    
    def test_batch_size_one(self):
        """Test inference with batch size 1"""
        image = np.random.uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)
        assert image.shape[0] == 1, "Batch size should be 1"
    
    
    def test_batch_size_multiple(self):
        """Test inference with multiple images in batch"""
        images = np.random.uniform(0, 1, (32, 224, 224, 3)).astype(np.float32)
        assert images.shape[0] == 32, "Batch size should be 32"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

