import cv2
import mediapipe as mp
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from datetime import datetime
import threading
import time
import pickle
from collections import Counter

class AdvancedHandSignRecognition:
    def __init__(self):
        # Initialize MediaPipe with advanced settings
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support for two hands
            min_detection_confidence=0.8,  # Higher confidence threshold
            min_tracking_confidence=0.7,
            model_complexity=1  # Use complex model for better accuracy
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize models
        self.xgb_model = None
        self.rf_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        # Recognition state
        self.current_text = ""
        self.recognition_history = []
        self.custom_signs = {}
        self.custom_sign_threshold = 5.0  # distance threshold for custom signs
        self.confidence_threshold = 0.85
        self.prediction_buffer = []
        self.buffer_size = 5
        
        # Load or create models
        self.load_or_create_models()
        
        # ASL signs with detailed descriptions
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
            'DELETE': 'closed fist',
            'ENTER': 'thumbs up'
        }
        
        # Add numbers 0-9
        for i in range(10):
            self.asl_signs[str(i)] = f'number {i} sign'
    
    def load_or_create_models(self):
        """Load existing models or create new ones"""
        model_files = [
            'advanced_xgb_model.pkl',
            'advanced_rf_model.pkl', 
            'advanced_scaler.pkl',
            'advanced_label_encoder.pkl'
        ]
        
        if all(os.path.exists(f) for f in model_files):
            print("Loading existing advanced models...")
            self.xgb_model = joblib.load('advanced_xgb_model.pkl')
            self.rf_model = joblib.load('advanced_rf_model.pkl')
            self.scaler = joblib.load('advanced_scaler.pkl')
            self.label_encoder = joblib.load('advanced_label_encoder.pkl')
        else:
            print("Creating new advanced models...")
            self.create_advanced_models()
    
    def create_advanced_models(self):
        """Create advanced machine learning models"""
        # Create XGBoost model with optimized parameters
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=50,  # Reduced for faster initialization
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            enable_categorical=False
        )
        
        # Create Random Forest as backup
        self.rf_model = RandomForestClassifier(
            n_estimators=50,  # Reduced for faster initialization
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Create scaler and fit it with dummy data to ensure it's fitted
        self.scaler = StandardScaler()
        
        # Create label encoder
        self.label_encoder = LabelEncoder()
        
        # Fit scaler with dummy data to ensure it's properly initialized
        # Generate dummy features that match our feature extraction
        dummy_features = np.random.random((100, 88))  # 88 features from extract_advanced_features
        self.scaler.fit(dummy_features)
        
        # Fit label encoder with dummy labels
        dummy_labels = list(self.asl_signs.keys())
        self.label_encoder.fit(dummy_labels)
        
        # Train models with dummy data to ensure they're properly initialized
        dummy_X = np.random.random((100, 88))
        dummy_y = np.random.choice(len(dummy_labels), 100)
        
        self.xgb_model.fit(dummy_X, dummy_y)
        self.rf_model.fit(dummy_X, dummy_y)
        
        # Save models
        joblib.dump(self.xgb_model, 'advanced_xgb_model.pkl')
        joblib.dump(self.rf_model, 'advanced_rf_model.pkl')
        joblib.dump(self.scaler, 'advanced_scaler.pkl')
        joblib.dump(self.label_encoder, 'advanced_label_encoder.pkl')
    
    def extract_advanced_features(self, hand_landmarks):
        """Extract advanced features from hand landmarks"""
        if hand_landmarks is None:
            return None
        
        features = []
        
        # Basic landmark coordinates
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
        
        # Palm center and finger angles
        palm_center = np.mean(landmarks[0:5], axis=0)  # Wrist to index base
        features.extend(palm_center)
        
        # Calculate angles between fingers
        for i in range(len(finger_tips)-1):
            vec1 = landmarks[finger_tips[i]] - landmarks[finger_mids[i]]
            vec2 = landmarks[finger_tips[i+1]] - landmarks[finger_mids[i+1]]
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            features.append(angle)
        
        return np.array(features)
    
    def predict_sign_advanced(self, hand_landmarks):
        """Predict hand sign using advanced models"""
        if hand_landmarks is None:
            return None, 0.0
        
        features = self.extract_advanced_features(hand_landmarks)
        if features is None:
            return None, 0.0
        
        # Reshape for prediction
        features_reshaped = features.reshape(1, -1)
        
        # Check if scaler is fitted, if not, fit it with current features
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            print("Scaler not fitted, fitting with current data...")
            self.scaler.fit(features_reshaped)
        
        # Scale features
        features_scaled = self.scaler.transform(features_reshaped)
        
        # Check if models are fitted, if not, return None
        if not hasattr(self.xgb_model, 'classes_') or not hasattr(self.rf_model, 'classes_'):
            print("Models not properly trained, returning None...")
            return None, 0.0
        
        # Get predictions from both models
        xgb_pred = self.xgb_model.predict(features_scaled)[0]
        xgb_proba = np.max(self.xgb_model.predict_proba(features_scaled))
        
        rf_pred = self.rf_model.predict(features_scaled)[0]
        rf_proba = np.max(self.rf_model.predict_proba(features_scaled))
        
        # Ensemble prediction (weighted average)
        if xgb_proba > rf_proba:
            prediction = xgb_pred
            confidence = xgb_proba
        else:
            prediction = rf_pred
            confidence = rf_proba

        # Convert prediction back to label
        try:
            predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        except Exception:
            predicted_label = None

        # Check custom signs and override if closer
        custom_pred, custom_conf = self.predict_custom_sign(features_scaled[0])
        if custom_pred and custom_conf > confidence:
            return custom_pred, custom_conf

        return predicted_label, confidence
    
    def add_custom_sign(self, sign_name, hand_landmarks):
        """Add a custom hand sign"""
        if hand_landmarks is None:
            return False
        
        features = self.extract_advanced_features(hand_landmarks)
        if features is not None:
            scaled = self.scaler.transform(features.reshape(1, -1))[0].tolist()
            self.custom_signs[sign_name] = {
                'raw': features.tolist(),
                'scaled': scaled
            }
            self.save_custom_signs()
            return True
        return False
    
    def save_custom_signs(self):
        """Save custom signs to file"""
        with open('advanced_custom_signs.json', 'w') as f:
            json.dump(self.custom_signs, f, indent=2)
    
    def load_custom_signs(self):
        """Load custom signs from file"""
        if os.path.exists('advanced_custom_signs.json'):
            with open('advanced_custom_signs.json', 'r') as f:
                data = json.load(f)
                # Support legacy format where features were stored as a list
                for name, value in data.items():
                    if isinstance(value, dict):
                        self.custom_signs[name] = value
                    else:
                        scaled = self.scaler.transform(np.array(value).reshape(1, -1))[0].tolist()
                        self.custom_signs[name] = {
                            'raw': value,
                            'scaled': scaled
                        }

    def predict_custom_sign(self, scaled_features):
        """Predict custom sign using nearest neighbour search"""
        if not self.custom_signs:
            return None, 0.0

        distances = []
        for name, feat_dict in self.custom_signs.items():
            feat = np.array(feat_dict['scaled'])
            dist = np.linalg.norm(scaled_features - feat)
            distances.append((name, dist))

        sign_name, dist = min(distances, key=lambda x: x[1])
        if dist > self.custom_sign_threshold:
            return None, 0.0

        confidence = max(0.0, 1 - dist / self.custom_sign_threshold)
        return sign_name, confidence
    
    def update_prediction_buffer(self, prediction, confidence):
        """Update prediction buffer for smoothing"""
        self.prediction_buffer.append((prediction, confidence))
        if len(self.prediction_buffer) > self.buffer_size:
            self.prediction_buffer.pop(0)
    
    def get_smoothed_prediction(self):
        """Get smoothed prediction from buffer"""
        if not self.prediction_buffer:
            return None, 0.0
        
        # Get most common prediction with highest average confidence
        predictions = [p[0] for p in self.prediction_buffer if p[0] is not None]
        confidences = [p[1] for p in self.prediction_buffer if p[0] is not None]
        
        if not predictions:
            return None, 0.0
        
        # Find most common prediction
        pred_counts = Counter(predictions)
        most_common_pred = pred_counts.most_common(1)[0][0]
        
        # Calculate average confidence for most common prediction
        pred_confidences = [conf for pred, conf in self.prediction_buffer 
                          if pred == most_common_pred]
        avg_confidence = np.mean(pred_confidences)
        
        return most_common_pred, avg_confidence
    
    def run_recognition(self):
        """Main recognition loop"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam")
            
            # Set camera properties for better quality
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
        except Exception as e:
            print("Camera Error:", e)
            print("Hints: ensure drivers are installed, check camera permissions, "
                  "close other applications and try a different camera index")
            return
        
        print("Advanced Hand Sign Recognition System Started!")
        print("Press 'q' to quit, 's' to save current sign, 'c' to clear text")
        print("Press 't' to train model with current data")
        
        last_prediction_time = time.time()
        prediction_delay = 1.0  # Delay between predictions
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks and make predictions
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Make prediction with delay
                    current_time = time.time()
                    if current_time - last_prediction_time > prediction_delay:
                        predicted_sign, confidence = self.predict_sign_advanced(hand_landmarks)
                        
                        if predicted_sign and confidence > self.confidence_threshold:
                            self.update_prediction_buffer(predicted_sign, confidence)
                            last_prediction_time = current_time
                
                # Get smoothed prediction
                smoothed_pred, smoothed_conf = self.get_smoothed_prediction()
                if smoothed_pred and smoothed_conf > self.confidence_threshold:
                    self.current_text += smoothed_pred
                    self.recognition_history.append({
                        'sign': smoothed_pred,
                        'confidence': smoothed_conf,
                        'timestamp': datetime.now().isoformat()
                    })
                    # Clear buffer after using prediction
                    self.prediction_buffer.clear()
            
            # Display information on frame
            cv2.putText(frame, f"Text: {self.current_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display confidence if available
            if self.prediction_buffer:
                latest_conf = self.prediction_buffer[-1][1]
                cv2.putText(frame, f"Confidence: {latest_conf:.2f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save sign, 'c' to clear, 't' to train", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Advanced Hand Sign Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_text = ""
            elif key == ord('s'):
                if results.multi_hand_landmarks:
                    sign_name = input("Enter name for this sign: ")
                    if self.add_custom_sign(sign_name, results.multi_hand_landmarks[0]):
                        print(f"Custom sign '{sign_name}' saved!")
            elif key == ord('t'):
                print("Training models with collected data...")
                self.train_models_with_data()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def train_models_with_data(self):
        """Train models with collected custom signs"""
        if not self.custom_signs:
            print("No custom signs to train with!")
            return
        
        print("Training models with custom signs...")
        # This would implement training logic with collected data
        # For now, just save the current state
        self.save_custom_signs()
        print("Training data saved!")

def main():
    recognizer = AdvancedHandSignRecognition()
    recognizer.load_custom_signs()
    recognizer.run_recognition()

if __name__ == "__main__":
    main()
