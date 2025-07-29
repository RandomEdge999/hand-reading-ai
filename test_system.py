#!/usr/bin/env python3
"""
Test script for Hand Sign Recognition System
Verifies that all components are working properly
"""

import sys
import os
import importlib
import subprocess

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    required_packages = [
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('PIL', 'Pillow'),
        ('joblib', 'Joblib')
    ]
    
    failed_imports = []
    
    for package, name in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✓ {name} imported successfully")
            elif package == 'mediapipe':
                import mediapipe
                print(f"✓ {name} imported successfully")
            elif package == 'tensorflow':
                import tensorflow
                print(f"✓ {name} imported successfully")
            elif package == 'numpy':
                import numpy
                print(f"✓ {name} imported successfully")
            elif package == 'sklearn':
                import sklearn
                print(f"✓ {name} imported successfully")
            elif package == 'matplotlib':
                import matplotlib
                print(f"✓ {name} imported successfully")
            elif package == 'PIL':
                from PIL import Image
                print(f"✓ {name} imported successfully")
            elif package == 'joblib':
                import joblib
                print(f"✓ {name} imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {name}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_camera():
    """Test if camera can be accessed"""
    print("\nTesting camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Camera not accessible")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print(f"✓ Camera working - Frame size: {frame.shape}")
            return True
        else:
            print("✗ Camera not working properly")
            return False
            
    except Exception as e:
        print(f"✗ Camera test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe hand detection"""
    print("\nTesting MediaPipe hand detection...")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Create a dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process image
        results = hands.process(dummy_image)
        
        print("✓ MediaPipe hand detection initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def test_tensorflow():
    """Test TensorFlow model creation"""
    print("\nTesting TensorFlow model creation...")
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        import numpy as np
        
        # Create a simple model
        model = keras.Sequential([
            keras.layers.Input(shape=(63,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(36, activation='softmax')
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        # Test prediction
        dummy_input = np.random.random((1, 63))
        prediction = model.predict(dummy_input, verbose=0)
        
        print(f"✓ TensorFlow model created successfully - Output shape: {prediction.shape}")
        return True
        
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        return False

def test_files():
    """Test if all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        'hand_sign_recognition.py',
        'gui_app.py',
        'train_model.py',
        'run_system.py',
        'config.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def test_system_modules():
    """Test if system modules can be imported"""
    print("\nTesting system modules...")
    
    system_modules = [
        'hand_sign_recognition',
        'gui_app',
        'train_model',
        'run_system',
        'config'
    ]
    
    failed_modules = []
    
    for module in system_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}.py imported successfully")
        except ImportError as e:
            print(f"✗ Failed to import {module}.py: {e}")
            failed_modules.append(module)
    
    return len(failed_modules) == 0

def run_comprehensive_test():
    """Run all tests"""
    print("=" * 50)
    print("Hand Sign Recognition System - Comprehensive Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Camera Access", test_camera),
        ("MediaPipe Hand Detection", test_mediapipe),
        ("TensorFlow Model Creation", test_tensorflow),
        ("File Structure", test_files),
        ("System Modules", test_system_modules)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nTo start the system:")
        print("1. Run: python run_system.py")
        print("2. Or double-click: start_system.bat")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check camera permissions")
        print("3. Ensure all files are in the same directory")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 