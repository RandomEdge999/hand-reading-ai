import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import queue
import numpy as np
import json
import os
from datetime import datetime

class HandSignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Sign Recognition System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize video capture
        self.cap = None
        self.is_running = False
        self.video_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        # Recognition state
        self.current_text = ""
        self.recognition_history = []
        self.custom_signs = {}
        
        # Load custom signs
        self.load_custom_signs()
        
        # Create GUI
        self.create_widgets()
        
        # Start video thread
        self.start_video()
    
    def create_widgets(self):
        """Create the GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Hand Sign Recognition System", 
                               font=('Arial', 24, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Video frame
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.video_label = ttk.Label(video_frame, text="Initializing camera...")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Recognition text
        ttk.Label(control_frame, text="Recognized Text:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        self.text_display = tk.Text(control_frame, height=8, width=40, font=('Arial', 12))
        self.text_display.pack(fill=tk.X, pady=(0, 20))
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Clear Text", command=self.clear_text).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Save Custom Sign", command=self.save_custom_sign).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Train Model", command=self.open_training).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="View History", command=self.view_history).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Export Text", command=self.export_text).pack(fill=tk.X, pady=2)
        
        # Status
        self.status_label = ttk.Label(control_frame, text="Status: Ready", font=('Arial', 10))
        self.status_label.pack(anchor=tk.W, pady=(20, 0))
        
        # Custom signs list
        ttk.Label(control_frame, text="Custom Signs:", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=(20, 5))
        
        self.signs_listbox = tk.Listbox(control_frame, height=6, font=('Arial', 10))
        self.signs_listbox.pack(fill=tk.X)
        self.update_signs_list()
    
    def start_video(self):
        """Start video capture thread"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")
        except Exception as e:
            hint = (
                "Failed to access the camera.\n\n"
                "Troubleshooting tips:\n"
                "- Ensure webcam drivers are installed and up to date\n"
                "- Check camera permissions\n"
                "- Close other applications using the camera\n"
                "- Try a different camera index"
            )
            messagebox.showerror("Camera Error", f"{e}\n\n{hint}")
            return
        
        self.is_running = True
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
        
        self.update_video()
    
    def video_loop(self):
        """Video processing loop"""
        import mediapipe as mp
        
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Predict sign (simplified for demo)
                    predicted_sign = self.predict_sign_simple(hand_landmarks)
                    if predicted_sign:
                        self.result_queue.put(predicted_sign)
            
            # Add text overlay
            cv2.putText(frame, f"Text: {self.current_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Resize for display
            display_size = (640, 480)
            pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Put in queue
            try:
                self.video_queue.put_nowait(pil_image)
            except queue.Full:
                pass
    
    def update_video(self):
        """Update video display"""
        try:
            while not self.video_queue.empty():
                pil_image = self.video_queue.get_nowait()
                photo = ImageTk.PhotoImage(pil_image)
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo
        except queue.Empty:
            pass
        
        # Check for recognition results
        try:
            while not self.result_queue.empty():
                sign = self.result_queue.get_nowait()
                self.current_text += sign
                self.recognition_history.append({
                    'sign': sign,
                    'timestamp': datetime.now().isoformat()
                })
                self.update_text_display()
        except queue.Empty:
            pass
        
        if self.is_running:
            self.root.after(30, self.update_video)
    
    def predict_sign_simple(self, hand_landmarks):
        """Simple sign prediction (placeholder for actual ML model)"""
        # This is a simplified version - in practice, you'd use your trained model
        # For now, we'll just return a placeholder
        return None
    
    def clear_text(self):
        """Clear the recognized text"""
        self.current_text = ""
        self.update_text_display()
    
    def update_text_display(self):
        """Update the text display"""
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.current_text)
    
    def save_custom_sign(self):
        """Save current hand position as custom sign"""
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not available")
            return
        
        sign_name = tk.simpledialog.askstring("Save Custom Sign", "Enter name for this sign:")
        if sign_name:
            # In a real implementation, you'd capture the current hand landmarks
            # and save them to the custom_signs dictionary
            self.custom_signs[sign_name] = {
                'timestamp': datetime.now().isoformat(),
                'description': 'Custom hand sign'
            }
            self.save_custom_signs()
            self.update_signs_list()
            messagebox.showinfo("Success", f"Custom sign '{sign_name}' saved!")
    
    def load_custom_signs(self):
        """Load custom signs from file"""
        if os.path.exists('custom_signs.json'):
            with open('custom_signs.json', 'r') as f:
                self.custom_signs = json.load(f)
    
    def save_custom_signs(self):
        """Save custom signs to file"""
        with open('custom_signs.json', 'w') as f:
            json.dump(self.custom_signs, f)
    
    def update_signs_list(self):
        """Update the custom signs listbox"""
        self.signs_listbox.delete(0, tk.END)
        for sign_name in self.custom_signs.keys():
            self.signs_listbox.insert(tk.END, sign_name)
    
    def open_training(self):
        """Open the training window"""
        training_window = tk.Toplevel(self.root)
        training_window.title("Model Training")
        training_window.geometry("600x400")
        
        # Training interface would go here
        ttk.Label(training_window, text="Training Interface", font=('Arial', 16, 'bold')).pack(pady=20)
        ttk.Label(training_window, text="Use train_model.py for detailed training").pack()
    
    def view_history(self):
        """View recognition history"""
        history_window = tk.Toplevel(self.root)
        history_window.title("Recognition History")
        history_window.geometry("500x400")
        
        history_text = tk.Text(history_window, font=('Arial', 10))
        history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for entry in self.recognition_history:
            history_text.insert(tk.END, f"{entry['timestamp']}: {entry['sign']}\n")
    
    def export_text(self):
        """Export recognized text to file"""
        if not self.current_text:
            messagebox.showwarning("Warning", "No text to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                f.write(self.current_text)
            messagebox.showinfo("Success", f"Text exported to {filename}")
    
    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = HandSignGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 