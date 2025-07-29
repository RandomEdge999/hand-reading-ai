import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from datetime import datetime
import threading
import time

class BasicHandSignRecognition:
    def __init__(self):
        # Initialize OpenCV hand detection
        self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
        
        # Use skin color detection
        self.use_skin_detection = True
        
        # Initialize model and data
        self.model = None
        self.scaler = None
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
        model_path = 'hand_sign_model.pkl'
        scaler_path = 'hand_sign_scaler.pkl'
        encoder_path = 'label_encoder.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
            print("Loading existing model...")
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
        else:
            print("Creating new model...")
            self.create_initial_model()
    
    def create_initial_model(self):
        """Create a simple Random Forest classifier"""
        # Create a simple Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Create label encoder
        self.label_encoder = {chr(i): i-65 for i in range(65, 91)}  # A-Z
        self.label_encoder.update({str(i): i+26 for i in range(10)})  # 0-9
        
        # Save model
        joblib.dump(self.model, 'hand_sign_model.pkl')
        joblib.dump(self.scaler, 'hand_sign_scaler.pkl')
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
        hand_roi = cv2.resize(hand_roi, (32, 32))
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        
        # Flatten and normalize
        features = gray_roi.flatten().astype(np.float32) / 255.0
        
        return features.reshape(1, -1)
    
    def predict_sign(self, hand_features):
        """Predict hand sign from features"""
        if hand_features is None or self.model is None:
            return None
        
        try:
            # Scale features
            scaled_features = self.scaler.transform(hand_features)
            
            # Predict
            prediction = self.model.predict(scaled_features)
            predicted_class = prediction[0]
            
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
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open webcam")
        except Exception as e:
            print("Camera Error:", e)
            print("Hints: ensure drivers are installed, check camera permissions, "
                  "close other applications and try a different camera index")
            return
        
        print("Basic Hand Sign Recognition System Started!")
        print("Press 'q' to quit, 's' to save current sign, 'c' to clear text")
        print("Note: This is a basic version without pre-trained model")
        
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
                    
                    # Extract features
                    hand_features = self.extract_hand_features(frame, hand_contour)
                    
                    # For demo purposes, show that hand is detected
                    cv2.putText(frame, "Hand Detected!", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Note: Prediction requires training data
                    # predicted_sign = self.predict_sign(hand_features)
                    # if predicted_sign:
                    #     self.current_text += predicted_sign
                
                # Show skin mask (for debugging)
                cv2.imshow('Skin Mask', skin_mask)
            
            # Display current text
            cv2.putText(frame, f"Text: {self.current_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save sign, 'c' to clear", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Basic Hand Sign Recognition', frame)
            
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
    recognizer = BasicHandSignRecognition()
    recognizer.load_custom_signs()
    recognizer.run_recognition()

if __name__ == "__main__":
    main() 