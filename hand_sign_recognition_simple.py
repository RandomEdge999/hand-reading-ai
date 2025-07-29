import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import json
from datetime import datetime
import threading
import time

class SimpleHandSignRecognition:
    def __init__(self):
        # Initialize OpenCV hand detection
        self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
        
        # Alternative: use skin color detection
        self.use_skin_detection = True
        
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
        # Simple CNN model for image-based classification
        self.model = keras.Sequential([
            keras.layers.Input(shape=(64, 64, 3)),  # 64x64 RGB image
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
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
    
    def detect_skin(self, frame):
        """Detect skin color in the frame"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        
        return mask
    
    def find_hand_contour(self, mask):
        """Find the largest contour (likely the hand)"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Filter by area to avoid noise
            if area > 5000:  # Minimum area threshold
                return largest_contour
        
        return None
    
    def extract_hand_features(self, frame, hand_contour):
        """Extract features from hand contour"""
        if hand_contour is None:
            return None
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(hand_contour)
        
        # Extract hand region
        hand_roi = frame[y:y+h, x:x+w]
        
        if hand_roi.size == 0:
            return None
        
        # Resize to standard size
        hand_roi = cv2.resize(hand_roi, (64, 64))
        
        # Normalize pixel values
        hand_roi = hand_roi.astype(np.float32) / 255.0
        
        return hand_roi.reshape(1, 64, 64, 3)
    
    def predict_sign(self, hand_features):
        """Predict hand sign from features"""
        if hand_features is None:
            return None
        
        try:
            prediction = self.model.predict(hand_features, verbose=0)
            predicted_class = np.argmax(prediction[0])
            
            # Convert back to label
            for label, idx in self.label_encoder.items():
                if idx == predicted_class:
                    return label
            
        except Exception as e:
            print(f"Prediction error: {e}")
        
        return None
    
    def add_custom_sign(self, sign_name, hand_features):
        """Add a custom hand sign for learning"""
        if hand_features is None:
            return False
        
        self.custom_signs[sign_name] = hand_features.flatten()
        
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
        
        print("Simple Hand Sign Recognition System Started!")
        print("Press 'q' to quit, 's' to save current sign, 'c' to clear text")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand using skin color
            if self.use_skin_detection:
                skin_mask = self.detect_skin(frame)
                hand_contour = self.find_hand_contour(skin_mask)
                
                # Draw hand contour
                if hand_contour is not None:
                    cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
                    
                    # Extract features and predict
                    hand_features = self.extract_hand_features(frame, hand_contour)
                    predicted_sign = self.predict_sign(hand_features)
                    
                    if predicted_sign:
                        self.current_text += predicted_sign
                        self.recognition_history.append({
                            'sign': predicted_sign,
                            'timestamp': datetime.now().isoformat()
                        })
                
                # Show skin mask (for debugging)
                cv2.imshow('Skin Mask', skin_mask)
            
            # Display current text
            cv2.putText(frame, f"Text: {self.current_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save sign, 'c' to clear", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Simple Hand Sign Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_text = ""
            elif key == ord('s'):
                if hand_contour is not None:
                    hand_features = self.extract_hand_features(frame, hand_contour)
                    sign_name = input("Enter name for this sign: ")
                    if self.add_custom_sign(sign_name, hand_features):
                        print(f"Custom sign '{sign_name}' saved!")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    recognizer = SimpleHandSignRecognition()
    recognizer.load_custom_signs()
    recognizer.run_recognition()

if __name__ == "__main__":
    main() 