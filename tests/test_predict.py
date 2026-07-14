#Import the libraries
import numpy as np
import pickle

def test_model_loads():
    # Test that model loads successfully
    with open('models/car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert model is not None
    
def test_model_prediction():
    # Test that model returns a prediction
    with open('models/car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    sample = np.array([[2018, 9.85, 30000, 2, 0, 1, 0]])
    prediction = model.predict(sample)
    
    assert prediction is not None
    assert len(prediction) == 1
    assert prediction[0] > 0

def test_prediction_range():
    # Test that prediction is in reasonable range
    with open('models/car_price_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    sample = np.array([[2018, 9.85, 30000, 2, 0, 1, 0]])
    prediction = model.predict(sample)
    
    assert 0 < prediction[0] < 50
    
    