import cv2
import mediapipe as mp
import numpy as np
import os
import json
from datetime import datetime
import threading
import time

class HandSignTrainer:
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
        
        # Training data storage
        self.training_data = {}
        self.current_sign = None
        self.sample_count = 0
        self.max_samples = 50
        
        # Load existing training data
        self.load_training_data()
    
    def load_training_data(self):
        """Load existing training data"""
        if os.path.exists('training_data.json'):
            with open('training_data.json', 'r') as f:
                self.training_data = json.load(f)
    
    def save_training_data(self):
        """Save training data to file"""
        with open('training_data.json', 'w') as f:
            json.dump(self.training_data, f)
    
    def extract_hand_features(self, hand_landmarks):
        """Extract features from hand landmarks"""
        features = []
        for landmark in hand_landmarks.landmark:
            features.extend([landmark.x, landmark.y, landmark.z])
        return features
    
    def collect_samples(self, sign_name):
        """Collect training samples for a specific sign"""
        self.current_sign = sign_name
        self.sample_count = 0
        
        if sign_name not in self.training_data:
            self.training_data[sign_name] = []
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print(f"Collecting samples for sign '{sign_name}'")
        print("Press 's' to save sample, 'q' to quit collection")
        
        while self.sample_count < self.max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
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
            
            # Display info
            cv2.putText(frame, f"Sign: {sign_name}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {self.sample_count}/{self.max_samples}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to save sample", 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "Press 'q' to quit", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Training Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if results.multi_hand_landmarks:
                    features = self.extract_hand_features(results.multi_hand_landmarks[0])
                    self.training_data[sign_name].append({
                        'features': features,
                        'timestamp': datetime.now().isoformat()
                    })
                    self.sample_count += 1
                    print(f"Sample {self.sample_count} saved!")
        
        cap.release()
        cv2.destroyAllWindows()
        self.save_training_data()
        print(f"Collection complete! {self.sample_count} samples saved for '{sign_name}'")
    
    def interactive_training(self):
        """Interactive training session"""
        print("=== Hand Sign Training System ===")
        print("Available commands:")
        print("1. 'collect <sign_name>' - Collect samples for a sign")
        print("2. 'list' - List all trained signs")
        print("3. 'delete <sign_name>' - Delete a sign")
        print("4. 'quit' - Exit training")
        
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command.startswith('collect '):
                sign_name = command[8:].upper()
                self.collect_samples(sign_name)
            
            elif command == 'list':
                if self.training_data:
                    print("\nTrained signs:")
                    for sign, samples in self.training_data.items():
                        print(f"  {sign}: {len(samples)} samples")
                else:
                    print("No signs trained yet.")
            
            elif command.startswith('delete '):
                sign_name = command[7:].upper()
                if sign_name in self.training_data:
                    del self.training_data[sign_name]
                    self.save_training_data()
                    print(f"Sign '{sign_name}' deleted.")
                else:
                    print(f"Sign '{sign_name}' not found.")
            
            elif command == 'quit':
                break
            
            else:
                print("Invalid command. Use 'collect <sign>', 'list', 'delete <sign>', or 'quit'")

def main():
    trainer = HandSignTrainer()
    trainer.interactive_training()

if __name__ == "__main__":
    main() 