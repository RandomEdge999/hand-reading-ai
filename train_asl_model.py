#!/usr/bin/env python3
"""
ASL Model Training Script
Trains the hand sign recognition models using real ASL alphabet images
"""

import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
from datetime import datetime
import glob
from collections import Counter

class ASLModelTrainer:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,  # Use static mode for image processing
            max_num_hands=1,
            min_detection_confidence=0.3,  # Lower threshold for better detection
            model_complexity=1
        )
        
        # Initialize models
        self.xgb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Training data
        self.X = []
        self.y = []
        
        # ASL signs mapping
        self.asl_signs = {
            'A': 'thumb out, fingers closed',
            'B': 'all fingers extended, thumb tucked',
            'C': 'curved hand like letter C',
            'D': 'index finger up, others closed',
            'E': 'all fingers closed, thumb across palm',
            'F': 'index and thumb touching, other fingers up',
            'G': 'index finger pointing',
            'H': 'index and middle finger extended',
            'I': 'pinky finger up, others closed',
            'J': 'pinky finger moving in J motion',
            'K': 'index and middle finger up, thumb between',
            'L': 'thumb and index finger forming L',
            'M': 'three fingers down, thumb tucked',
            'N': 'two fingers down, thumb tucked',
            'O': 'all fingers curved together',
            'P': 'index finger pointing down',
            'Q': 'index finger pointing down, thumb out',
            'R': 'index and middle finger crossed',
            'S': 'fist with thumb over fingers',
            'T': 'index finger up, thumb touching middle',
            'U': 'index and middle finger together up',
            'V': 'index and middle finger apart up',
            'W': 'three fingers up (thumb, index, middle)',
            'X': 'index finger bent',
            'Y': 'thumb and pinky out',
            'Z': 'index finger moving in Z motion',
            'SPACE': 'open palm facing forward',
            'NOTHING': 'no hand detected'
        }
    
    def extract_features(self, hand_landmarks):
        """Extract features from hand landmarks"""
        if hand_landmarks is None:
            return None
        
        features = []
        
        # Basic landmark coordinates (21 landmarks * 3 coordinates = 63 features)
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        
        # Calculate distances between key points
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # Hand size features
        hand_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        hand_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        hand_depth = np.max(landmarks[:, 2]) - np.min(landmarks[:, 2])
        
        features.extend([hand_width, hand_height, hand_depth])
        
        # Finger length ratios
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, index, middle, ring, pinky
        finger_mids = [3, 7, 11, 15, 19]
        finger_bases = [2, 6, 10, 14, 18]
        
        for tip, mid, base in zip(finger_tips, finger_mids, finger_bases):
            # Calculate finger lengths
            tip_to_mid = np.linalg.norm(landmarks[tip] - landmarks[mid])
            mid_to_base = np.linalg.norm(landmarks[mid] - landmarks[base])
            total_length = tip_to_mid + mid_to_base
            
            features.extend([tip_to_mid, mid_to_base, total_length])
        
        # Palm center
        palm_center = np.mean(landmarks[0:5], axis=0)  # Wrist to index base
        features.extend(palm_center)
        
        # Calculate angles between fingers
        for i in range(len(finger_tips)-1):
            vec1 = landmarks[finger_tips[i]] - landmarks[finger_mids[i]]
            vec2 = landmarks[finger_tips[i+1]] - landmarks[finger_mids[i+1]]
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            features.append(angle)
        
        return np.array(features)
    
    def load_and_process_images(self, folder_path="asl_alphabet_test"):
        """Load and process ASL images from the folder"""
        print("Loading and processing ASL images...")
        
        # Get all jpg files in the folder
        image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return False
        
        processed_count = 0
        failed_count = 0
        
        for image_file in image_files:
            # Extract label from filename (e.g., "A_test.jpg" -> "A")
            filename = os.path.basename(image_file)
            label = filename.split('_')[0].upper()
            
            # Handle special cases
            if label == "SPACE":
                label = "SPACE"
            elif label == "NOTHING":
                label = "NOTHING"
            
            # Skip if label not in our ASL signs
            if label not in self.asl_signs:
                print(f"Skipping {filename} - unknown label: {label}")
                continue
            
            # Load image
            image = cv2.imread(image_file)
            if image is None:
                print(f"Failed to load image: {image_file}")
                failed_count += 1
                continue
            
            # Preprocess image to improve hand detection
            # Resize if too small
            height, width = image.shape[:2]
            if width < 300 or height < 300:
                scale = max(300/width, 300/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
            
            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.hands.process(rgb_image)
            
            if results.multi_hand_landmarks:
                # Use the first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Extract features
                features = self.extract_features(hand_landmarks)
                
                if features is not None:
                    self.X.append(features)
                    self.y.append(label)
                    processed_count += 1
                    print(f"✓ Processed {filename} -> {label}")
                else:
                    print(f"✗ Failed to extract features from {filename}")
                    failed_count += 1
            else:
                # For "nothing" images, create a dummy feature vector
                if label == "NOTHING":
                    dummy_features = np.zeros(88)  # 88 features
                    self.X.append(dummy_features)
                    self.y.append(label)
                    processed_count += 1
                    print(f"✓ Processed {filename} -> {label} (no hand detected)")
                else:
                    print(f"✗ No hand detected in {filename}")
                    failed_count += 1
        
        print(f"\nProcessing complete:")
        print(f"✓ Successfully processed: {processed_count} images")
        print(f"✗ Failed to process: {failed_count} images")
        
        return len(self.X) > 0
    
    def train_models(self):
        """Train XGBoost and Random Forest models"""
        if not self.X or not self.y:
            print("No training data available!")
            return False
        
        print(f"\nTraining models with {len(self.X)} samples...")
        
        # Convert to numpy arrays
        X = np.array(self.X)
        y = np.array(self.y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Ensure all classes are present in training data
        unique_classes = np.unique(y_encoded)
        if len(unique_classes) != len(self.label_encoder.classes_):
            print(f"Warning: Only {len(unique_classes)} out of {len(self.label_encoder.classes_)} classes have samples")
            print(f"Available classes: {unique_classes}")
            print(f"All classes: {list(self.label_encoder.classes_)}")
        
        # For small datasets, train on all data
        if len(X) < 50:
            print("Small dataset detected, training on all data")
            X_train = X
            y_train = y_encoded
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Create a small validation set for testing
            X_val = X[:2]  # Use first 2 samples for validation
            y_val = y_encoded[:2]
            X_val_scaled = self.scaler.transform(X_val)
        else:
            # Split data (without stratification if we have too few samples per class)
            if len(np.unique(y_encoded)) >= 2 and all(np.bincount(y_encoded) >= 2):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            else:
                print("Warning: Not enough samples per class for stratified split, using random split")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.2, random_state=42
                )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            X_val_scaled = X_test_scaled
            y_val = y_test
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        
        # Train XGBoost
        print("\nTraining XGBoost model...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            enable_categorical=False
        )
        
        # Ensure all classes are present in training data for XGBoost
        all_classes = np.arange(len(self.label_encoder.classes_))
        missing_classes = set(all_classes) - set(np.unique(y_train))
        
        if missing_classes:
            print(f"Adding dummy samples for missing classes: {missing_classes}")
            # Add dummy samples for missing classes
            dummy_features = np.zeros((len(missing_classes), X_train_scaled.shape[1]))
            dummy_labels = list(missing_classes)
            
            X_train_extended = np.vstack([X_train_scaled, dummy_features])
            y_train_extended = np.concatenate([y_train, dummy_labels])
            
            self.xgb_model.fit(X_train_extended, y_train_extended)
        else:
            self.xgb_model.fit(X_train_scaled, y_train)
        xgb_pred = self.xgb_model.predict(X_val_scaled)
        xgb_accuracy = accuracy_score(y_val, xgb_pred)
        
        print(f"XGBoost accuracy: {xgb_accuracy:.4f}")
        
        # Train Random Forest
        print("\nTraining Random Forest model...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.rf_model.fit(X_train_scaled, y_train)
        rf_pred = self.rf_model.predict(X_val_scaled)
        rf_accuracy = accuracy_score(y_val, rf_pred)
        
        print(f"Random Forest accuracy: {rf_accuracy:.4f}")
        
        # Print simple accuracy report
        print(f"\nXGBoost accuracy: {xgb_accuracy:.4f}")
        print(f"Random Forest accuracy: {rf_accuracy:.4f}")
        
        # Store accuracies for saving
        self.xgb_accuracy = xgb_accuracy
        self.rf_accuracy = rf_accuracy
        
        return True
    
    def save_models(self):
        """Save trained models"""
        if not self.xgb_model or not self.rf_model:
            print("Models not trained yet!")
            return False
        
        print("\nSaving models...")
        
        # Save models
        joblib.dump(self.xgb_model, 'advanced_xgb_model.pkl')
        joblib.dump(self.rf_model, 'advanced_rf_model.pkl')
        joblib.dump(self.scaler, 'advanced_scaler.pkl')
        joblib.dump(self.label_encoder, 'advanced_label_encoder.pkl')
        
        # Save training info
        training_info = {
            'training_date': datetime.now().isoformat(),
            'num_samples': len(self.X),
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'feature_count': len(self.X[0]) if self.X else 0,
            'xgb_accuracy': getattr(self, 'xgb_accuracy', 'N/A'),
            'rf_accuracy': getattr(self, 'rf_accuracy', 'N/A')
        }
        
        with open('training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print("✓ Models saved successfully!")
        print(f"✓ Training info saved to training_info.json")
        
        return True
    
    def run_training(self):
        """Run the complete training pipeline"""
        print("=" * 60)
        print("ASL HAND SIGN RECOGNITION MODEL TRAINING")
        print("=" * 60)
        
        # Step 1: Load and process images
        if not self.load_and_process_images():
            print("Failed to load training data!")
            return False
        
        # Step 2: Train models
        if not self.train_models():
            print("Failed to train models!")
            return False
        
        # Step 3: Save models
        if not self.save_models():
            print("Failed to save models!")
            return False
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print("Models are ready to use with the advanced hand recognition system.")
        
        return True

def main():
    """Main function"""
    trainer = ASLModelTrainer()
    success = trainer.run_training()
    
    if success:
        print("\nYou can now run the system with:")
        print("python working_launcher.py")
    else:
        print("\nTraining failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
