#!/usr/bin/env python3
"""
Basic Hand Sign Recognition System Launcher
Works with Python 3.13 and available packages
"""

import sys
import subprocess
import os
import tkinter as tk
from tkinter import ttk, messagebox

class BasicSystemLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Basic Hand Sign Recognition System")
        self.root.geometry("500x400")
        self.root.configure(bg='#2c3e50')
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create the launcher interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Basic Hand Sign Recognition", 
                               font=('Arial', 20, 'bold'))
        title_label.pack(pady=(0, 30))
        
        # Description
        desc_label = ttk.Label(main_frame, 
                              text="Python 3.13 Compatible Version\n(Simplified for compatibility)",
                              font=('Arial', 12))
        desc_label.pack(pady=(0, 20))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        # Basic Recognition
        basic_btn = ttk.Button(button_frame, text="🎥 Basic Hand Detection", 
                              command=self.start_basic_recognition)
        basic_btn.pack(fill=tk.X, pady=5)
        
        # Test System
        test_btn = ttk.Button(button_frame, text="🔧 Test System", 
                             command=self.test_system)
        test_btn.pack(fill=tk.X, pady=5)
        
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
                                     font=('Arial', 10))
        self.status_label.pack(anchor=tk.W, pady=(20, 0))
    
    def start_basic_recognition(self):
        """Start the basic hand recognition system"""
        self.status_label.config(text="Starting basic recognition...")
        self.root.update()
        
        try:
            subprocess.Popen([sys.executable, "hand_sign_recognition_basic.py"])
            self.status_label.config(text="Basic recognition started!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start basic recognition:\n{str(e)}")
            self.status_label.config(text="Failed to start basic recognition")
    
    def test_system(self):
        """Test the system components"""
        self.status_label.config(text="Testing system...")
        self.root.update()
        
        try:
            # Test basic imports
            import cv2
            import numpy as np
            from sklearn.ensemble import RandomForestClassifier
            from PIL import Image
            
            messagebox.showinfo("System Test", "All basic components working!\n\n" +
                               "✓ OpenCV (cv2)\n" +
                               "✓ NumPy\n" +
                               "✓ Scikit-learn\n" +
                               "✓ Pillow (PIL)\n\n" +
                               "System is ready to use!")
            self.status_label.config(text="System test passed")
            
        except ImportError as e:
            messagebox.showerror("System Test Failed", 
                               f"Missing component: {str(e)}\n\n" +
                               "Please run: pip install -r requirements.txt")
            self.status_label.config(text="System test failed")
    
    def check_dependencies(self):
        """Check if all dependencies are installed"""
        self.status_label.config(text="Checking dependencies...")
        self.root.update()
        
        required_packages = [
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy'),
            ('sklearn', 'Scikit-learn'),
            ('PIL', 'Pillow'),
            ('joblib', 'Joblib')
        ]
        
        missing_packages = []
        
        for package, name in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'numpy':
                    import numpy
                elif package == 'sklearn':
                    import sklearn
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'joblib':
                    import joblib
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            messagebox.showerror("Missing Dependencies", 
                               f"The following packages are missing:\n{', '.join(missing_packages)}\n\n"
                               "Please run: pip install -r requirements.txt")
            self.status_label.config(text="Missing dependencies detected")
        else:
            messagebox.showinfo("Dependencies Check", "All required packages are installed!")
            self.status_label.config(text="All dependencies are installed")
    
    def run(self):
        """Run the launcher"""
        self.root.mainloop()

def main():
    """Main function"""
    print("Basic Hand Sign Recognition System Launcher")
    print("Python 3.13 Compatible Version")
    print("Starting launcher...")
    
    launcher = BasicSystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main() 