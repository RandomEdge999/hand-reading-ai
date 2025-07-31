#!/usr/bin/env python3
"""
Working Launcher for Advanced Hand Sign Recognition System
Bypasses any hanging issues and provides a working system
"""

import sys
import subprocess
import os
import tkinter as tk
from tkinter import ttk, messagebox

class WorkingSystemLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Working Hand Sign Recognition System")
        self.root.geometry("600x500")
        self.root.configure(bg='#1e1e1e')
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create the launcher interface"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Hand Sign Recognition System", 
                               font=('Arial', 24, 'bold'), foreground='#00ff00')
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, text="Local AI-Powered System", 
                                 font=('Arial', 12), foreground='#cccccc')
        subtitle_label.pack(pady=(0, 30))
        
        # Description
        desc_frame = ttk.LabelFrame(main_frame, text="System Features", padding=10)
        desc_frame.pack(fill=tk.X, pady=(0, 20))
        
        features = [
            "🎯 Advanced MediaPipe hand tracking",
            "🧠 XGBoost + Random Forest ensemble models",
            "📊 Real-time confidence scoring",
            "🎨 Modern GUI interface",
            "💾 Local model training and storage",
            "📱 High-resolution camera support"
        ]
        
        for feature in features:
            ttk.Label(desc_frame, text=feature, font=('Arial', 10)).pack(anchor=tk.W, pady=2)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        # Test System First
        test_btn = ttk.Button(button_frame, text="🔧 Test System First", 
                             command=self.test_system, style='Accent.TButton')
        test_btn.pack(fill=tk.X, pady=5)
        
        # Launch GUI
        gui_btn = ttk.Button(button_frame, text="🎥 Launch GUI Application", 
                            command=self.start_gui)
        gui_btn.pack(fill=tk.X, pady=5)
        
        # Launch CLI
        cli_btn = ttk.Button(button_frame, text="💻 Launch Command Line", 
                            command=self.start_cli)
        cli_btn.pack(fill=tk.X, pady=5)
        
        # Check Dependencies
        deps_btn = ttk.Button(button_frame, text="📦 Check Dependencies", 
                             command=self.check_dependencies)
        deps_btn.pack(fill=tk.X, pady=5)
        
        # Exit button
        exit_btn = ttk.Button(button_frame, text="❌ Exit", 
                             command=self.root.quit)
        exit_btn.pack(fill=tk.X, pady=5)
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Ready to launch", 
                                     font=('Arial', 10), foreground='#00ff00')
        self.status_label.pack(anchor=tk.W, pady=(20, 0))
    
    def test_system(self):
        """Test the system components"""
        self.status_label.config(text="Testing system...", foreground='#ffff00')
        self.root.update()
        
        try:
            # Test basic imports
            import cv2
            import mediapipe
            import numpy
            import xgboost
            import sklearn
            
            # Test camera
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    camera_status = "working"
                else:
                    camera_status = "not working properly"
            else:
                camera_status = "not accessible"
            
            # Test ML models
            import xgboost as xgb
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            
            X = np.random.random((26, 63))
            y = np.arange(26)
            
            xgb_model = xgb.XGBClassifier(n_estimators=5, random_state=42, eval_metric='mlogloss', enable_categorical=False)
            rf_model = RandomForestClassifier(n_estimators=5, random_state=42)
            
            xgb_model.fit(X, y)
            rf_model.fit(X, y)
            
            messagebox.showinfo("System Test", 
                              f"All components working!\n\n" +
                              "✓ OpenCV (cv2)\n" +
                              "✓ MediaPipe\n" +
                              "✓ XGBoost\n" +
                              "✓ Random Forest\n" +
                              "✓ Scikit-learn\n" +
                              f"✓ Camera: {camera_status}\n\n" +
                              "System is ready to use!")
            self.status_label.config(text="System test passed", foreground='#00ff00')
            
        except ImportError as e:
            messagebox.showerror("Import Error", f"Missing package: {e}\n\nPlease run: pip install -r requirements.txt")
            self.status_label.config(text="Import error", foreground='#ff0000')
        except Exception as e:
            messagebox.showerror("Test Error", f"Test failed: {e}")
            self.status_label.config(text="Test failed", foreground='#ff0000')
    
    def start_gui(self):
        """Start the GUI application"""
        self.status_label.config(text="Starting GUI application...", foreground='#ffff00')
        self.root.update()
        
        try:
            # Try to start the advanced GUI, fallback to basic if needed
            if os.path.exists('advanced_gui_app.py'):
                subprocess.Popen([sys.executable, "advanced_gui_app.py"])
                self.status_label.config(text="Advanced GUI started!", foreground='#00ff00')
            elif os.path.exists('gui_app.py'):
                subprocess.Popen([sys.executable, "gui_app.py"])
                self.status_label.config(text="Basic GUI started!", foreground='#00ff00')
            else:
                messagebox.showerror("Error", "No GUI application found!")
                self.status_label.config(text="No GUI found", foreground='#ff0000')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start GUI:\n{str(e)}")
            self.status_label.config(text="Failed to start GUI", foreground='#ff0000')
    
    def start_cli(self):
        """Start the command line application"""
        self.status_label.config(text="Starting CLI application...", foreground='#ffff00')
        self.root.update()
        
        try:
            # Try to start the advanced CLI, fallback to basic if needed
            if os.path.exists('advanced_hand_recognition.py'):
                subprocess.Popen([sys.executable, "advanced_hand_recognition.py"])
                self.status_label.config(text="Advanced CLI started!", foreground='#00ff00')
            elif os.path.exists('hand_sign_recognition.py'):
                subprocess.Popen([sys.executable, "hand_sign_recognition.py"])
                self.status_label.config(text="Basic CLI started!", foreground='#00ff00')
            else:
                messagebox.showerror("Error", "No CLI application found!")
                self.status_label.config(text="No CLI found", foreground='#ff0000')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start CLI:\n{str(e)}")
            self.status_label.config(text="Failed to start CLI", foreground='#ff0000')
    
    def check_dependencies(self):
        """Check if all dependencies are installed"""
        self.status_label.config(text="Checking dependencies...", foreground='#ffff00')
        self.root.update()
        
        required_packages = [
            ('cv2', 'OpenCV'),
            ('mediapipe', 'MediaPipe'),
            ('numpy', 'NumPy'),
            ('xgboost', 'XGBoost'),
            ('sklearn', 'Scikit-learn'),
            ('PIL', 'Pillow'),
            ('joblib', 'Joblib')
        ]
        
        missing_packages = []
        
        for package, name in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'mediapipe':
                    import mediapipe
                elif package == 'numpy':
                    import numpy
                elif package == 'xgboost':
                    import xgboost
                elif package == 'sklearn':
                    import sklearn
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'joblib':
                    import joblib
            except ImportError:
                missing_packages.append(name)
        
        if missing_packages:
            messagebox.showerror("Missing Dependencies", 
                               f"The following packages are missing:\n{', '.join(missing_packages)}\n\n"
                               "Please run: pip install -r requirements.txt")
            self.status_label.config(text="Missing dependencies", foreground='#ff0000')
        else:
            messagebox.showinfo("Dependencies Check", "All packages are installed!")
            self.status_label.config(text="All dependencies installed", foreground='#00ff00')
    
    def run(self):
        """Run the launcher"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Working Hand Sign Recognition System Launcher")
    print("Starting launcher...")
    
    launcher = WorkingSystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
