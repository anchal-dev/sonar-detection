"""
Test script for Sonar Detection API
Run this after starting the Flask server to verify all endpoints
"""

import requests
import json
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

API_URL = "http://localhost:5000/api"

def print_header(text):
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.CYAN}{text:^60}")
    print(f"{Fore.CYAN}{'='*60}\n")

def print_success(text):
    print(f"{Fore.GREEN}✓ {text}")

def print_error(text):
    print(f"{Fore.RED}✗ {text}")

def print_info(text):
    print(f"{Fore.YELLOW}ℹ {text}")

def test_health_check():
    """Test the health check endpoint"""
    print_header("Testing Health Check Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        data = response.json()
        
        print_success(f"Status Code: {response.status_code}")
        print_success(f"API Status: {data['status']}")
        print_success(f"Model Loaded: {data['model_loaded']}")
        print_success(f"Version: {data['version']}")
        
        return data['model_loaded']
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False

def test_stats_endpoint():
    """Test the statistics endpoint"""
    print_header("Testing Statistics Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        data = response.json()
        
        print_success(f"Status Code: {response.status_code}")
        print_info(f"Training Accuracy: {data['training_accuracy']}%")
        print_info(f"Test Accuracy: {data['test_accuracy']}%")
        print_info(f"Total Samples: {data['total_samples']}")
        print_info(f"Rock Samples: {data['rock_samples']}")
        print_info(f"Mine Samples: {data['mine_samples']}")
        
        return True
    except Exception as e:
        print_error(f"Stats endpoint failed: {str(e)}")
        return False

def test_sample_endpoint(sample_type):
    """Test the sample data endpoint"""
    print_header(f"Testing Sample Endpoint - {sample_type.upper()}")
    
    try:
        response = requests.get(f"{API_URL}/sample/{sample_type}", timeout=5)
        data = response.json()
        
        print_success(f"Status Code: {response.status_code}")
        print_info(f"Sample Type: {data['type']}")
        print_info(f"Number of Features: {len(data['features'])}")
        print_info(f"Expected Label: {data['label']}")
        print_info(f"First 5 values: {data['features'][:5]}")
        
        return data
    except Exception as e:
        print_error(f"Sample endpoint failed: {str(e)}")
        return None

def test_prediction(sample_type):
    """Test the prediction endpoint"""
    print_header(f"Testing Prediction - {sample_type.upper()} Sample")
    
    # First get the sample data
    sample_data = test_sample_endpoint(sample_type)
    
    if not sample_data:
        return False
    
    try:
        # Make prediction
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": sample_data['features']},
            timeout=5
        )
        data = response.json()
        
        print_success(f"Status Code: {response.status_code}")
        print_success(f"Prediction: {data['prediction']} ({data['prediction_label']})")
        print_info(f"Message: {data['message']}")
        print_info(f"Rock Confidence: {data['confidence']['rock']}%")
        print_info(f"Mine Confidence: {data['confidence']['mine']}%")
        
        # Verify prediction matches expected label
        if data['prediction'] == sample_data['label']:
            print_success(f"✓ Prediction matches expected label!")
        else:
            print_error(f"✗ Prediction ({data['prediction']}) doesn't match expected ({sample_data['label']})")
        
        return True
    except Exception as e:
        print_error(f"Prediction failed: {str(e)}")
        return False

def test_invalid_input():
    """Test API error handling with invalid input"""
    print_header("Testing Error Handling")
    
    test_cases = [
        {
            "name": "Empty features",
            "data": {"features": []},
            "expected_status": 400
        },
        {
            "name": "Wrong number of features",
            "data": {"features": [0.1, 0.2, 0.3]},
            "expected_status": 400
        },
        {
            "name": "Missing features key",
            "data": {},
            "expected_status": 400
        },
        {
            "name": "Non-numeric values",
            "data": {"features": ["a"] * 60},
            "expected_status": 400
        }
    ]
    
    for test in test_cases:
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json=test["data"],
                timeout=5
            )
            
            if response.status_code == test["expected_status"]:
                print_success(f"{test['name']}: Correctly returned {response.status_code}")
            else:
                print_error(f"{test['name']}: Expected {test['expected_status']}, got {response.status_code}")
        except Exception as e:
            print_error(f"{test['name']}: {str(e)}")

def test_custom_input():
    """Test with custom random input"""
    print_header("Testing Custom Random Input")
    
    import random
    
    # Generate random features
    random_features = [round(random.random() * 0.5, 4) for _ in range(60)]
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": random_features},
            timeout=5
        )
        data = response.json()
        
        print_success(f"Status Code: {response.status_code}")
        print_info(f"Random features (first 5): {random_features[:5]}")
        print_success(f"Prediction: {data['prediction']} ({data['prediction_label']})")
        print_info(f"Rock Confidence: {data['confidence']['rock']}%")
        print_info(f"Mine Confidence: {data['confidence']['mine']}%")
        
        return True
    except Exception as e:
        print_error(f"Custom input test failed: {str(e)}")
        return False

def run_all_tests():
    """Run all API tests"""
    print(f"\n{Fore.MAGENTA}{'*'*60}")
    print(f"{Fore.MAGENTA}{'SONAR DETECTION API TEST SUITE':^60}")
    print(f"{Fore.MAGENTA}{'*'*60}\n")
    
    results = {
        "Health Check": test_health_check(),
        "Statistics": test_stats_endpoint(),
        "Rock Sample": test_sample_endpoint("rock") is not None,
        "Mine Sample": test_sample_endpoint("mine") is not None,
        "Rock Prediction": test_prediction("rock"),
        "Mine Prediction": test_prediction("mine"),
        "Custom Input": test_custom_input(),
    }
    
    # Test error handling
    test_invalid_input()
    
    # Summary
    print_header("Test Summary")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = f"{Fore.GREEN}PASSED" if result else f"{Fore.RED}FAILED"
        print(f"{test_name:.<40}{status}")
    
    print(f"\n{Fore.CYAN}{'─'*60}")
    percentage = (passed / total) * 100
    color = Fore.GREEN if percentage == 100 else Fore.YELLOW if percentage >= 70 else Fore.RED
    print(f"{color}Total: {passed}/{total} tests passed ({percentage:.1f}%)")
    print(f"{Fore.CYAN}{'─'*60}\n")
    
    if percentage == 100:
        print(f"{Fore.GREEN}🎉 All tests passed! Your API is ready for presentation! 🎉")
    elif percentage >= 70:
        print(f"{Fore.YELLOW}⚠️  Most tests passed. Check failed tests above.")
    else:
        print(f"{Fore.RED}❌ Several tests failed. Please check your setup.")

if __name__ == "__main__":
    print(f"\n{Fore.CYAN}Starting API tests...")
    print(f"{Fore.CYAN}Make sure Flask server is running on http://localhost:5000\n")
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Tests interrupted by user")
    except Exception as e:
        print(f"\n{Fore.RED}Test suite error: {str(e)}")