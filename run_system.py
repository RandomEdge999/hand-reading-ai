#!/usr/bin/env python3
"""
Hand Sign Recognition System Launcher
A simple launcher to start different components of the system.
"""

import sys
import subprocess
import os
import tkinter as tk
from tkinter import ttk, messagebox

class SystemLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hand Sign Recognition System Launcher")
        self.root.geometry("500x400")
        self.root.configure(bg='#2c3e50')
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create the launcher interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Hand Sign Recognition System", 
                               font=('Arial', 20, 'bold'))
        title_label.pack(pady=(0, 30))
        
        # Description
        desc_label = ttk.Label(main_frame, 
                              text="Choose an option to start the system:",
                              font=('Arial', 12))
        desc_label.pack(pady=(0, 20))
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        # GUI Application (Recommended)
        gui_btn = ttk.Button(button_frame, text="🎥 GUI Application (Recommended)", 
                            command=self.start_gui, style='Accent.TButton')
        gui_btn.pack(fill=tk.X, pady=5)
        
        # Command Line Application
        cli_btn = ttk.Button(button_frame, text="💻 Command Line Application", 
                            command=self.start_cli)
        cli_btn.pack(fill=tk.X, pady=5)
        
        # Training System
        train_btn = ttk.Button(button_frame, text="🎓 Training System", 
                              command=self.start_training)
        train_btn.pack(fill=tk.X, pady=5)
        
        # Check Dependencies
        deps_btn = ttk.Button(button_frame, text="🔧 Check Dependencies", 
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
    
    def start_gui(self):
        """Start the GUI application"""
        self.status_label.config(text="Starting GUI application...")
        self.root.update()
        
        try:
            subprocess.Popen([sys.executable, "gui_app.py"])
            self.status_label.config(text="GUI application started successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start GUI application:\n{str(e)}")
            self.status_label.config(text="Failed to start GUI application")
    
    def start_cli(self):
        """Start the command line application"""
        self.status_label.config(text="Starting CLI application...")
        self.root.update()
        
        try:
            subprocess.Popen([sys.executable, "hand_sign_recognition.py"])
            self.status_label.config(text="CLI application started successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start CLI application:\n{str(e)}")
            self.status_label.config(text="Failed to start CLI application")
    
    def start_training(self):
        """Start the training system"""
        self.status_label.config(text="Starting training system...")
        self.root.update()
        
        try:
            subprocess.Popen([sys.executable, "train_model.py"])
            self.status_label.config(text="Training system started successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training system:\n{str(e)}")
            self.status_label.config(text="Failed to start training system")
    
    def check_dependencies(self):
        """Check if all dependencies are installed"""
        self.status_label.config(text="Checking dependencies...")
        self.root.update()
        
        required_packages = [
            'cv2',
            'mediapipe',
            'tensorflow',
            'numpy',
            'sklearn',
            'matplotlib',
            'PIL'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'cv2':
                    import cv2
                elif package == 'mediapipe':
                    import mediapipe
                elif package == 'tensorflow':
                    import tensorflow
                elif package == 'numpy':
                    import numpy
                elif package == 'sklearn':
                    import sklearn
                elif package == 'matplotlib':
                    import matplotlib
                elif package == 'PIL':
                    from PIL import Image
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
    print("Hand Sign Recognition System Launcher")
    print("Starting launcher...")
    
    launcher = SystemLauncher()
    launcher.run()

if __name__ == "__main__":
    main() 