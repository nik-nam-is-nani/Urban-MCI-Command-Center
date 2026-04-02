"""Test script to verify Flask API works correctly."""
import json
from app import app

def test_api():
    with app.test_client() as client:
        # Test /reset
        resp = client.post('/reset', json={'task': 1})
        print(f'POST /reset: {resp.status_code}')
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = json.loads(resp.data)
        print(f'  Task: {data.get("task")}')
        print(f'  State keys: {list(data.get("state", {}).keys())[:5]}')
        
        # Test /step
        resp = client.post('/step', json={'directives': []})
        print(f'POST /step: {resp.status_code}')
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        
        # Test /grade
        resp = client.get('/grade')
        print(f'GET /grade: {resp.status_code}')
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        data = json.loads(resp.data)
        print(f'  Grade: {data.get("grade")}')
        
        # Test /health
        resp = client.get('/health')
        print(f'GET /health: {resp.status_code}')
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        
        # Test /tasks
        resp = client.get('/tasks')
        print(f'GET /tasks: {resp.status_code}')
        
        print('All API tests passed!')

if __name__ == '__main__':
    test_api()