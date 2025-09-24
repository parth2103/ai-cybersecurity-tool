#!/usr/bin/env python3
"""
API endpoint tests for AI Cybersecurity Tool
Simple tests that can run without the full integration test suite
"""

import pytest
import requests
import json
import time

@pytest.mark.api
class TestAPIEndpoints:
    """Test API endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup(self, api_base_url):
        self.base_url = api_base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.session.get(f"{self.base_url}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = self.session.get(f"{self.base_url}/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert 'total_requests' in data
        assert 'threats_detected' in data
        assert 'threat_history' in data
        assert 'current_threat_level' in data
        
        # Check data types
        assert isinstance(data['total_requests'], int)
        assert isinstance(data['threats_detected'], int)
        assert isinstance(data['threat_history'], list)
        assert isinstance(data['current_threat_level'], str)
    
    def test_system_info_endpoint(self):
        """Test system information endpoint"""
        response = self.session.get(f"{self.base_url}/system/info")
        assert response.status_code == 200
        
        data = response.json()
        assert 'models_loaded' in data
        assert 'cpu_percent' in data
        assert 'memory_percent' in data
        assert 'total_predictions' in data
        assert 'threats_detected' in data
        assert 'detection_rate' in data
        
        # Check data types and ranges
        assert isinstance(data['models_loaded'], list)
        assert 0 <= data['cpu_percent'] <= 100
        assert 0 <= data['memory_percent'] <= 100
        assert isinstance(data['total_predictions'], int)
        assert isinstance(data['threats_detected'], int)
        assert isinstance(data['detection_rate'], (int, float))
    
    def test_alerts_endpoint(self):
        """Test alerts endpoint"""
        response = self.session.get(f"{self.base_url}/alerts")
        assert response.status_code == 200
        
        data = response.json()
        assert 'alerts' in data
        assert isinstance(data['alerts'], list)
    
    def test_predict_endpoint_structure(self, sample_request_data):
        """Test prediction endpoint response structure"""
        response = self.session.post(
            f"{self.base_url}/predict",
            json=sample_request_data
        )
        assert response.status_code == 200
        
        data = response.json()
        assert 'threat_detected' in data
        assert 'threat_level' in data
        assert 'threat_score' in data
        assert 'model_predictions' in data
        assert 'timestamp' in data
        
        # Check data types
        assert isinstance(data['threat_detected'], bool)
        assert isinstance(data['threat_level'], str)
        assert isinstance(data['threat_score'], (int, float))
        assert isinstance(data['model_predictions'], dict)
        assert isinstance(data['timestamp'], str)
        
        # Check threat score range
        assert 0 <= data['threat_score'] <= 1
        
        # Check threat level is valid
        assert data['threat_level'] in ['None', 'Low', 'Medium', 'High', 'Critical']
    
    def test_predict_endpoint_performance(self, sample_request_data):
        """Test prediction endpoint performance"""
        start_time = time.time()
        response = self.session.post(
            f"{self.base_url}/predict",
            json=sample_request_data
        )
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed_time < 5.0, f"Response time {elapsed_time:.2f}s exceeds 5s requirement"
    
    def test_batch_predict_endpoint(self, sample_features):
        """Test batch prediction endpoint"""
        batch_data = {
            'logs': [
                {
                    **sample_features,
                    'source_ip': f'192.168.1.{i}',
                    'attack_type': 'test'
                }
                for i in range(1, 4)
            ]
        }
        
        response = self.session.post(
            f"{self.base_url}/batch/predict",
            json=batch_data
        )
        assert response.status_code == 200
        
        data = response.json()
        assert 'results' in data
        assert 'summary' in data
        assert len(data['results']) == 3
        
        # Check summary structure
        summary = data['summary']
        assert 'total_processed' in summary
        assert 'threats_detected' in summary
        assert 'average_threat_score' in summary
        assert 'critical_threats' in summary
        
        # Check data types
        assert isinstance(summary['total_processed'], int)
        assert isinstance(summary['threats_detected'], int)
        assert isinstance(summary['average_threat_score'], (int, float))
        assert isinstance(summary['critical_threats'], int)
    
    def test_invalid_request_handling(self):
        """Test handling of invalid requests"""
        # Test with missing features
        invalid_data = {
            'source_ip': '192.168.1.100',
            'attack_type': 'test'
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=invalid_data
        )
        
        # Should handle gracefully
        assert response.status_code in [400, 500]
    
    def test_model_loading(self):
        """Test that models are properly loaded"""
        response = self.session.get(f"{self.base_url}/system/info")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data['models_loaded']) > 0, "No models are loaded"
        assert 'rf' in data['models_loaded'], "Random Forest model not loaded"

@pytest.mark.performance
class TestAPIPerformance:
    """Test API performance under load"""
    
    @pytest.fixture(autouse=True)
    def setup(self, api_base_url, sample_request_data):
        self.base_url = api_base_url
        self.sample_data = sample_request_data
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def test_concurrent_requests(self):
        """Test system under concurrent load"""
        from concurrent.futures import ThreadPoolExecutor
        import numpy as np
        
        def make_request():
            # Modify source IP for each request
            data = self.sample_data.copy()
            data['source_ip'] = f'192.168.1.{np.random.randint(1, 255)}'
            
            return self.session.post(
                f"{self.base_url}/predict",
                json=data
            )
        
        # Make 10 concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in futures]
        
        # All should succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200, f"Request {i} failed with status {response.status_code}"
    
    def test_response_time_consistency(self):
        """Test that response times are consistent"""
        response_times = []
        
        for _ in range(5):
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/predict",
                json=self.sample_data
            )
            elapsed_time = time.time() - start_time
            
            assert response.status_code == 200
            response_times.append(elapsed_time)
        
        # Check that response times are consistent (within 2x of average)
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        
        assert max_time < avg_time * 2, f"Response time inconsistency: max={max_time:.2f}s, avg={avg_time:.2f}s"
