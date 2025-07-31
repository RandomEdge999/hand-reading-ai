#!/usr/bin/env python3
"""
Advanced Hand Sign Recognition System Test
Comprehensive testing for the local AI-powered hand sign recognition system
"""

import sys
import os
import importlib
import subprocess
import time

def test_advanced_imports():
    """Test if all advanced packages can be imported"""
    print("Testing advanced package imports...")
    
    required_packages = [
        ('cv2', 'OpenCV'),
        ('mediapipe', 'MediaPipe'),
        ('numpy', 'NumPy'),
        ('xgboost', 'XGBoost'),
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
            elif package == 'numpy':
                import numpy
                print(f"✓ {name} imported successfully")
            elif package == 'xgboost':
                import xgboost
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

def test_mediapipe_advanced():
    """Test MediaPipe advanced hand detection"""
    print("\nTesting MediaPipe advanced hand detection...")
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Initialize MediaPipe with advanced settings
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        # Create a dummy image
        dummy_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Process image
        results = hands.process(dummy_image)
        
        print("✓ MediaPipe advanced hand detection initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ MediaPipe advanced test failed: {e}")
        return False

def test_xgboost_models():
    """Test XGBoost model creation and training"""
    print("\nTesting XGBoost model creation...")
    
    try:
        import xgboost as xgb
        import numpy as np
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        # Create XGBoost model
        xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            enable_categorical=False
        )
        
        # Create dummy data with consistent classes
        X = np.random.random((100, 63))  # 63 features (21 landmarks * 3 coordinates)
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] * 4)[:100]  # 26 classes (A-Z)
        
        # Create scaler and encoder
        scaler = StandardScaler()
        encoder = LabelEncoder()
        
        # Scale features
        X_scaled = scaler.fit_transform(X)
        y_encoded = encoder.fit_transform(y)
        
        # Train model
        xgb_model.fit(X_scaled, y_encoded)
        
        # Test prediction
        prediction = xgb_model.predict(X_scaled[:1])
        probability = xgb_model.predict_proba(X_scaled[:1])
        
        print(f"✓ XGBoost model created and trained successfully - Output shape: {probability.shape}")
        return True
        
    except Exception as e:
        print(f"✗ XGBoost test failed: {e}")
        return False

def test_ensemble_models():
    """Test ensemble model creation"""
    print("\nTesting ensemble model creation...")
    
    try:
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Create XGBoost model with updated parameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=10, 
            random_state=42, 
            eval_metric='mlogloss',
            enable_categorical=False
        )
        
        # Create Random Forest model
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy data with consistent classes
        X = np.random.random((50, 63))
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] * 2)[:50]  # Ensure all classes are present
        
        # Train both models
        xgb_model.fit(X, y)
        rf_model.fit(X, y)
        
        # Test ensemble prediction
        xgb_pred = xgb_model.predict(X[:1])
        rf_pred = rf_model.predict(X[:1])
        
        print("✓ Ensemble models (XGBoost + Random Forest) created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Ensemble model test failed: {e}")
        return False

def test_advanced_files():
    """Test if all advanced system files exist"""
    print("\nTesting advanced file structure...")
    
    required_files = [
        'advanced_hand_recognition.py',
        'advanced_gui_app.py',
        'working_launcher.py',
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

def test_advanced_modules():
    """Test if advanced system modules can be imported"""
    print("\nTesting advanced system modules...")
    
    system_modules = [
        'advanced_hand_recognition',
        'advanced_gui_app',
        'working_launcher'
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

def test_advanced_model_prediction():
    """Test advanced model predictions"""
    print("\nTesting advanced model predictions...")
    
    try:
        from advanced_hand_recognition import AdvancedHandSignRecognition
        import numpy as np
        
        # Create recognizer instance
        recognizer = AdvancedHandSignRecognition()
        
        # Test feature extraction with dummy landmarks
        dummy_landmarks = type('Landmarks', (), {
            'landmark': [type('Landmark', (), {
                'x': np.random.random(),
                'y': np.random.random(),
                'z': np.random.random()
            })() for _ in range(21)]
        })()
        
        # Test feature extraction
        features = recognizer.extract_advanced_features(dummy_landmarks)
        
        if features is not None and len(features) > 0:
            print(f"✓ Advanced feature extraction working - Features: {len(features)}")
            return True
        else:
            print("✗ Advanced feature extraction failed")
            return False
            
    except Exception as e:
        print(f"✗ Advanced model prediction test failed: {e}")
        return False

def test_gui_components():
    """Test GUI components"""
    print("\nTesting GUI components...")
    
    try:
        import tkinter as tk
        from tkinter import ttk
        
        # Create test window
        root = tk.Tk()
        root.withdraw()  # Hide window
        
        # Test basic widgets
        frame = ttk.Frame(root)
        label = ttk.Label(frame, text="Test")
        button = ttk.Button(frame, text="Test")
        
        root.destroy()
        
        print("✓ GUI components working")
        return True
        
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        return False

def run_comprehensive_advanced_test():
    """Run all advanced system tests"""
    print("=" * 60)
    print("Advanced Hand Sign Recognition System - Comprehensive Test")
    print("=" * 60)
    
    tests = [
        ("Advanced Package Imports", test_advanced_imports),
        ("Camera Access", test_camera),
        ("MediaPipe Advanced Detection", test_mediapipe_advanced),
        ("XGBoost Model Creation", test_xgboost_models),
        ("Ensemble Models", test_ensemble_models),
        ("Advanced File Structure", test_advanced_files),
        ("Advanced System Modules", test_advanced_modules),
        ("Advanced Model Prediction", test_advanced_model_prediction),
        ("GUI Components", test_gui_components)
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
    print("\n" + "=" * 60)
    print("ADVANCED SYSTEM TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All advanced system tests passed! The system is ready to use.")
        print("\nTo start the advanced system:")
        print("1. Run: python working_launcher.py")
        print("2. Or run: python advanced_gui_app.py")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("\nCommon solutions:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Check camera permissions")
        print("3. Ensure all files are in the same directory")
        return False

if __name__ == "__main__":
    success = run_comprehensive_advanced_test()
    sys.exit(0 if success else 1) 