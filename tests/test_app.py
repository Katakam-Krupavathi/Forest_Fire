import os
import sys
import pytest

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Algerian Forest Fire Predictor' in response.data

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'healthy'
    assert json_data['model_loaded'] is True

def test_predict_page_get(client):
    response = client.get('/predict')
    assert response.status_code == 200
    assert b'Calculate Fire Weather Index' in response.data

def test_predict_form_post_success(client):
    payload = {
        'Temperature': '30',
        'RH': '60',
        'Ws': '15',
        'Rain': '0.0',
        'FFMC': '85.0',
        'DMC': '15.0',
        'ISI': '5.0',
        'Classes': '1',
        'Region': '0'
    }
    response = client.post('/predict', data=payload)
    assert response.status_code == 200
    assert b'Prediction Result' in response.data
    assert b'FWI:' in response.data

def test_predict_form_post_missing_field(client):
    payload = {
        'Temperature': '30',
        'RH': '60'
    }
    response = client.post('/predict', data=payload)
    assert response.status_code == 400
    assert b'Missing required parameter' in response.data

def test_predict_form_post_invalid_range(client):
    payload = {
        'Temperature': '30',
        'RH': '150',  # Invalid RH > 100
        'Ws': '15',
        'Rain': '0.0',
        'FFMC': '85.0',
        'DMC': '15.0',
        'ISI': '5.0',
        'Classes': '1',
        'Region': '0'
    }
    response = client.post('/predict', data=payload)
    assert response.status_code == 400
    assert b'Relative Humidity (RH) must be between 0% and 100%' in response.data

def test_predict_json_post_success(client):
    payload = {
        'Temperature': 32.5,
        'RH': 45.0,
        'Ws': 14.0,
        'Rain': 0.0,
        'FFMC': 89.0,
        'DMC': 20.0,
        'ISI': 7.5,
        'Classes': 1,
        'Region': 0
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['status'] == 'success'
    assert 'fwi' in json_data
    assert isinstance(json_data['fwi'], float)
    assert 'risk_level' in json_data

def test_predict_json_post_invalid(client):
    payload = {
        'Temperature': 32.5,
        'RH': -5.0
    }
    response = client.post('/predict', json=payload)
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data['status'] == 'error'
