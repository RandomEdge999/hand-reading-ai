import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import json
from datetime import datetime
import threading
import time

class HandSignRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize model and data
        self.model = None
        self.label_encoder = None
        self.custom_signs = {}
        self.current_text = ""
        self.recognition_history = []
        
        # Load or create model
        self.load_or_create_model()
        
        # ASL basic signs (A-Z, 0-9)
        self.basic_signs = {
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
            'Z': 'index finger moving in Z motion'
        }
        
        # Numbers 0-9
        for i in range(10):
            self.basic_signs[str(i)] = f'number {i} sign'
    
    def load_or_create_model(self):
        """Load existing model or create a new one"""
        model_path = 'hand_sign_model.h5'
        encoder_path = 'label_encoder.pkl'
        
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            print("Loading existing model...")
            self.model = keras.models.load_model(model_path)
            self.label_encoder = joblib.load(encoder_path)
        else:
            print("Creating new model...")
            self.create_initial_model()
    
    def create_initial_model(self):
        """Create a simple neural network for hand sign classification"""
        # Simple CNN model
        self.model = keras.Sequential([
            keras.layers.Input(shape=(21, 3)),  # 21 hand landmarks, 3 coordinates each
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(36, activation='softmax')  # 26 letters + 10 numbers
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create label encoder
        self.label_encoder = {chr(i): i-65 for i in range(65, 91)}  # A-Z
        self.label_encoder.update({str(i): i+26 for i in range(10)})  # 0-9
        
        # Save model
        self.model.save('hand_sign_model.h5')
        joblib.dump(self.label_encoder, 'label_encoder.pkl')
    
    def extract_hand_features(self, hand_landmarks):
        """Extract features from hand landmarks"""
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return np.array(features).reshape(1, -1)
    
    def predict_sign(self, hand_landmarks):
        """Predict hand sign from landmarks"""
        if hand_landmarks is None:
            return None
        
        features = self.extract_hand_features(hand_landmarks)
        prediction = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        # Convert back to label
        for label, idx in self.label_encoder.items():
            if idx == predicted_class:
                return label
        
        return None
    
    def add_custom_sign(self, sign_name, hand_landmarks):
        """Add a custom hand sign for learning"""
        if hand_landmarks is None:
            return False
        
        features = self.extract_hand_features(hand_landmarks)
        self.custom_signs[sign_name] = features.flatten()
        
        # Save custom signs
        self.save_custom_signs()
        return True
    
    def save_custom_signs(self):
        """Save custom signs to file"""
        with open('custom_signs.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in self.custom_signs.items()}, f)
    
    def load_custom_signs(self):
        """Load custom signs from file"""
        if os.path.exists('custom_signs.json'):
            with open('custom_signs.json', 'r') as f:
                data = json.load(f)
                self.custom_signs = {k: np.array(v) for k, v in data.items()}
    
    def run_recognition(self):
        """Main recognition loop"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Hand Sign Recognition System Started!")
        print("Press 'q' to quit, 's' to save current sign, 'c' to clear text")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Predict sign
                    predicted_sign = self.predict_sign(hand_landmarks)
                    if predicted_sign:
                        self.current_text += predicted_sign
                        self.recognition_history.append({
                            'sign': predicted_sign,
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Display current text
            cv2.putText(frame, f"Text: {self.current_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save sign, 'c' to clear", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Hand Sign Recognition', frame)
            
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
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = HandSignRecognition()
    recognizer.load_custom_signs()
    recognizer.run_recognition()

if __name__ == "__main__":
    main() 