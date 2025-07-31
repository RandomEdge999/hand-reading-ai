import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import queue
import numpy as np
import json
import os
import time
from datetime import datetime
import joblib
from advanced_hand_recognition import AdvancedHandSignRecognition

class AdvancedHandSignGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Hand Sign Recognition System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize video capture
        self.cap = None
        self.is_running = False
        self.video_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        
        # Recognition state
        self.current_text = ""
        self.recognition_history = []
        self.custom_signs = {}
        self.confidence_threshold = 0.85
        
        # Initialize advanced recognizer
        self.recognizer = AdvancedHandSignRecognition()
        self.recognizer.load_custom_signs()
        
        # Create GUI
        self.create_widgets()
        
        # Start video thread
        self.start_video()
    
    def create_widgets(self):
        """Create the advanced GUI widgets"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_label = ttk.Label(main_frame, text="Advanced Hand Sign Recognition System", 
                               font=('Arial', 28, 'bold'), foreground='#00ff00')
        title_label.pack(pady=(0, 20))
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        
        # Video frame
        video_frame = ttk.LabelFrame(left_panel, text="Live Camera Feed", padding=10)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_frame, text="Initializing camera...", 
                                   font=('Arial', 14))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Controls
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(15, 0))
        
        # Recognition text
        text_frame = ttk.LabelFrame(right_panel, text="Recognized Text", padding=10)
        text_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.text_display = tk.Text(text_frame, height=8, width=45, font=('Arial', 12),
                                  bg='#2d2d2d', fg='#ffffff', insertbackground='#ffffff')
        self.text_display.pack(fill=tk.X)
        
        # Control buttons
        control_frame = ttk.LabelFrame(right_panel, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Button grid
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X)
        
        # Row 1
        row1 = ttk.Frame(button_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(row1, text="🗑️ Clear Text", command=self.clear_text, 
                  style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(row1, text="💾 Save Sign", command=self.save_custom_sign,
                  style='Accent.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Row 2
        row2 = ttk.Frame(button_frame)
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Button(row2, text="🎓 Train Model", command=self.train_model,
                  style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(row2, text="📊 View History", command=self.view_history,
                  style='Accent.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Row 3
        row3 = ttk.Frame(button_frame)
        row3.pack(fill=tk.X, pady=2)
        
        ttk.Button(row3, text="📁 Export Text", command=self.export_text,
                  style='Accent.TButton').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(row3, text="⚙️ Settings", command=self.open_settings,
                  style='Accent.TButton').pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Status and info
        info_frame = ttk.LabelFrame(right_panel, text="System Information", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.status_label = ttk.Label(info_frame, text="Status: Ready", 
                                    font=('Arial', 10), foreground='#00ff00')
        self.status_label.pack(anchor=tk.W)
        
        self.confidence_label = ttk.Label(info_frame, text="Confidence: --", 
                                        font=('Arial', 10), foreground='#ffff00')
        self.confidence_label.pack(anchor=tk.W)
        
        # Custom signs list
        signs_frame = ttk.LabelFrame(right_panel, text="Custom Signs", padding=10)
        signs_frame.pack(fill=tk.BOTH, expand=True)
        
        self.signs_listbox = tk.Listbox(signs_frame, height=6, font=('Arial', 10),
                                      bg='#2d2d2d', fg='#ffffff', selectbackground='#404040')
        self.signs_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Signs control buttons
        signs_btn_frame = ttk.Frame(signs_frame)
        signs_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(signs_btn_frame, text="Delete", command=self.delete_custom_sign).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(signs_btn_frame, text="Refresh", command=self.update_signs_list).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Update signs list
        self.update_signs_list()
    
    def start_video(self):
        """Start video capture thread"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open webcam")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
            self.status_label.config(text="Status: Camera Active", foreground='#00ff00')
            
        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to start camera:\n{str(e)}")
            self.status_label.config(text="Status: Camera Error", foreground='#ff0000')
    
    def video_loop(self):
        """Video processing loop"""
        while self.is_running:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Process with MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.recognizer.hands.process(rgb_frame)
            
            # Draw landmarks and make predictions
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.recognizer.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.recognizer.mp_hands.HAND_CONNECTIONS,
                        self.recognizer.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.recognizer.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Make prediction
                    predicted_sign, confidence = self.recognizer.predict_sign_advanced(hand_landmarks)
                    
                    # Store current landmarks for custom sign saving
                    self.current_landmarks = hand_landmarks
                    
                    if predicted_sign and confidence > self.confidence_threshold:
                        self.recognizer.update_prediction_buffer(predicted_sign, confidence)
                        
                        # Update confidence display
                        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")
            
            # Get smoothed prediction
            smoothed_pred, smoothed_conf = self.recognizer.get_smoothed_prediction()
            if smoothed_pred and smoothed_conf > self.confidence_threshold:
                self.current_text += smoothed_pred
                self.recognition_history.append({
                    'sign': smoothed_pred,
                    'confidence': smoothed_conf,
                    'timestamp': datetime.now().isoformat()
                })
                self.recognizer.prediction_buffer.clear()
                self.update_text_display()
            
            # Add frame to queue
            try:
                self.video_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.video_queue.get_nowait()
                    self.video_queue.put_nowait(frame)
                except queue.Empty:
                    pass
            
            # Small delay
            cv2.waitKey(1)
    
    def update_video(self):
        """Update video display"""
        try:
            frame = self.video_queue.get_nowait()
            
            # Resize frame for display
            height, width = frame.shape[:2]
            max_width = 800
            max_height = 600
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo
            
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(30, self.update_video)
    
    def clear_text(self):
        """Clear recognized text"""
        self.current_text = ""
        self.update_text_display()
    
    def update_text_display(self):
        """Update text display"""
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(1.0, self.current_text)
    
    def save_custom_sign(self):
        """Save current hand position as custom sign"""
        # Get current landmarks from video processing
        if hasattr(self, 'current_landmarks') and self.current_landmarks is not None:
            sign_name = tk.simpledialog.askstring("Save Custom Sign", "Enter name for this sign:")
            if sign_name:
                if self.recognizer.add_custom_sign(sign_name, self.current_landmarks):
                    messagebox.showinfo("Success", f"Custom sign '{sign_name}' saved successfully!")
                    self.update_signs_list()
                else:
                    messagebox.showerror("Error", "Failed to save custom sign.")
        else:
            messagebox.showwarning("No Hand Detected", "Please show your hand to the camera first.")
    
    def load_custom_signs(self):
        """Load custom signs"""
        self.recognizer.load_custom_signs()
        self.custom_signs = self.recognizer.custom_signs
    
    def save_custom_signs(self):
        """Save custom signs"""
        self.recognizer.save_custom_signs()
    
    def update_signs_list(self):
        """Update custom signs list"""
        self.signs_listbox.delete(0, tk.END)
        for sign_name in self.recognizer.custom_signs.keys():
            self.signs_listbox.insert(tk.END, sign_name)
    
    def delete_custom_sign(self):
        """Delete selected custom sign"""
        selection = self.signs_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a sign to delete.")
            return
        
        sign_name = self.signs_listbox.get(selection[0])
        if messagebox.askyesno("Confirm Delete", f"Delete custom sign '{sign_name}'?"):
            if sign_name in self.recognizer.custom_signs:
                del self.recognizer.custom_signs[sign_name]
                self.recognizer.save_custom_signs()
                self.update_signs_list()
                messagebox.showinfo("Success", f"Sign '{sign_name}' deleted.")
    
    def train_model(self):
        """Train model with collected data"""
        if not self.recognizer.custom_signs:
            messagebox.showinfo("No Data", "No custom signs available for training.")
            return
        
        # Show training dialog
        training_window = tk.Toplevel(self.root)
        training_window.title("Model Training")
        training_window.geometry("400x300")
        
        ttk.Label(training_window, text="Training Model...", font=('Arial', 14)).pack(pady=20)
        
        progress = ttk.Progressbar(training_window, mode='indeterminate')
        progress.pack(pady=20, padx=20, fill=tk.X)
        progress.start()
        
        # Simulate training
        def train():
            time.sleep(2)  # Simulate training time
            self.recognizer.train_models_with_data()
            training_window.destroy()
            messagebox.showinfo("Training Complete", "Model training completed successfully!")
        
        threading.Thread(target=train, daemon=True).start()
    
    def view_history(self):
        """View recognition history"""
        if not self.recognition_history:
            messagebox.showinfo("No History", "No recognition history available.")
            return
        
        history_window = tk.Toplevel(self.root)
        history_window.title("Recognition History")
        history_window.geometry("600x400")
        
        # Create text widget for history
        history_text = tk.Text(history_window, font=('Arial', 10))
        history_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add history entries
        for entry in self.recognition_history[-50:]:  # Show last 50 entries
            timestamp = entry['timestamp'][:19]  # Remove microseconds
            history_text.insert(tk.END, f"[{timestamp}] {entry['sign']} (conf: {entry['confidence']:.2f})\n")
        
        history_text.config(state=tk.DISABLED)
    
    def export_text(self):
        """Export recognized text to file"""
        if not self.current_text:
            messagebox.showwarning("No Text", "No text to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.current_text)
                messagebox.showinfo("Export Success", f"Text exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {str(e)}")
    
    def open_settings(self):
        """Open settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        
        # Confidence threshold
        ttk.Label(settings_window, text="Confidence Threshold:", font=('Arial', 12)).pack(pady=10)
        
        confidence_var = tk.DoubleVar(value=self.confidence_threshold)
        confidence_scale = ttk.Scale(settings_window, from_=0.5, to=1.0, 
                                   variable=confidence_var, orient=tk.HORIZONTAL)
        confidence_scale.pack(pady=5, padx=20, fill=tk.X)
        
        confidence_label = ttk.Label(settings_window, text=f"{self.confidence_threshold:.2f}")
        confidence_label.pack()
        
        def update_confidence_label(value):
            confidence_label.config(text=f"{value:.2f}")
        
        confidence_scale.config(command=update_confidence_label)
        
        # Save button
        def save_settings():
            self.confidence_threshold = confidence_var.get()
            self.recognizer.confidence_threshold = confidence_var.get()
            settings_window.destroy()
            messagebox.showinfo("Settings Saved", "Settings saved successfully!")
        
        ttk.Button(settings_window, text="Save Settings", command=save_settings).pack(pady=20)
    
    def on_closing(self):
        """Handle window closing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

def main():
    """Main function"""
    root = tk.Tk()
    app = AdvancedHandSignGUI(root)
    
    # Start video updates
    app.update_video()
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    root.mainloop()

if __name__ == "__main__":
    main() 