import cv2
import numpy as np
import os
import json
from datetime import datetime

class HandSignDemo:
    def __init__(self):
        # Use skin color detection only (no cascade files needed)
        self.use_skin_detection = True
        
        # Initialize data
        self.custom_signs = {}
        self.current_text = ""
        self.recognition_history = []
        
        # Load custom signs
        self.load_custom_signs()
        
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
    
    def detect_skin(self, frame):
        """Detect skin color in the frame"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range (adjust for different lighting)
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
            if area > 3000:  # Lower threshold for better detection
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
        
        return features
    
    def add_custom_sign(self, sign_name, hand_features):
        """Add a custom hand sign for learning"""
        if hand_features is None:
            return False
        
        self.custom_signs[sign_name] = hand_features.tolist()
        
        # Save custom signs
        self.save_custom_signs()
        return True
    
    def save_custom_signs(self):
        """Save custom signs to file"""
        with open('custom_signs.json', 'w') as f:
            json.dump(self.custom_signs, f)
    
    def load_custom_signs(self):
        """Load custom signs from file"""
        if os.path.exists('custom_signs.json'):
            with open('custom_signs.json', 'r') as f:
                self.custom_signs = json.load(f)
    
    def run_demo(self):
        """Main demo loop"""
        # Try different camera indices
        camera_indices = [0, 1, -1]
        cap = None
        
        for idx in camera_indices:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Camera opened successfully with index {idx}")
                break
        
        if not cap or not cap.isOpened():
            print("Error: Could not open any camera")
            print("Please check your webcam connection and permissions")
            return
        
        print("Hand Sign Recognition Demo Started!")
        print("Press 'q' to quit, 's' to save current sign, 'c' to clear text")
        print("Show your hand to the camera for detection")
        
        frame_count = 0
        hand_detected_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand using skin color
            skin_mask = self.detect_skin(frame)
            hand_contour = self.find_hand_contour(skin_mask)
            
            # Draw hand contour and info
            if hand_contour is not None:
                cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
                cv2.putText(frame, "Hand Detected!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                hand_detected_frames += 1
                
                # Extract features for potential saving
                hand_features = self.extract_hand_features(frame, hand_contour)
                
                # Show hand area info
                area = cv2.contourArea(hand_contour)
                cv2.putText(frame, f"Area: {area:.0f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Hand Detected", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display current text
            cv2.putText(frame, f"Text: {self.current_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame count and detection rate
            frame_count += 1
            detection_rate = (hand_detected_frames / frame_count) * 100 if frame_count > 0 else 0
            cv2.putText(frame, f"Detection Rate: {detection_rate:.1f}%", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save sign, 'c' to clear", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show main frame
            cv2.imshow('Hand Sign Recognition Demo', frame)
            
            # Show skin mask (smaller window)
            skin_mask_resized = cv2.resize(skin_mask, (320, 240))
            cv2.imshow('Skin Detection', skin_mask_resized)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_text = ""
                print("Text cleared")
            elif key == ord('s'):
                if hand_contour is not None:
                    hand_features = self.extract_hand_features(frame, hand_contour)
                    sign_name = input("Enter name for this sign: ")
                    if self.add_custom_sign(sign_name, hand_features):
                        print(f"Custom sign '{sign_name}' saved!")
                        self.current_text += f"[{sign_name}]"
                else:
                    print("No hand detected to save")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\nDemo Summary:")
        print(f"Total frames processed: {frame_count}")
        print(f"Hand detection rate: {detection_rate:.1f}%")
        print(f"Custom signs saved: {len(self.custom_signs)}")

def main():
    print("=" * 50)
    print("Hand Sign Recognition Demo")
    print("Python 3.13 Compatible Version")
    print("=" * 50)
    
    demo = HandSignDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 