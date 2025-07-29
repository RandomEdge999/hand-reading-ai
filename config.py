"""
Configuration file for Hand Sign Recognition System
Contains all system settings and parameters
"""

import os

# Video Settings
VIDEO_SETTINGS = {
    'camera_index': 0,  # Default camera (0 for built-in webcam)
    'frame_width': 640,
    'frame_height': 480,
    'fps': 30,
    'flip_horizontal': True,  # Mirror effect
}

# MediaPipe Settings
MEDIAPIPE_SETTINGS = {
    'static_image_mode': False,
    'max_num_hands': 1,  # Number of hands to detect
    'min_detection_confidence': 0.7,  # Minimum confidence for hand detection
    'min_tracking_confidence': 0.5,   # Minimum confidence for hand tracking
}

# Model Settings
MODEL_SETTINGS = {
    'input_shape': (21, 3),  # 21 landmarks, 3 coordinates each
    'hidden_layers': [128, 64],  # Hidden layer sizes
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
}

# Training Settings
TRAINING_SETTINGS = {
    'samples_per_sign': 50,  # Number of samples to collect per sign
    'min_samples': 10,       # Minimum samples required for training
    'data_augmentation': True,
    'save_interval': 10,     # Save model every N epochs
}

# Recognition Settings
RECOGNITION_SETTINGS = {
    'confidence_threshold': 0.8,  # Minimum confidence for sign recognition
    'prediction_delay': 0.5,      # Delay between predictions (seconds)
    'smoothing_window': 5,        # Number of predictions to average
    'auto_clear_text': False,     # Auto-clear text after certain time
    'clear_delay': 10.0,          # Auto-clear delay (seconds)
}

# GUI Settings
GUI_SETTINGS = {
    'window_width': 1200,
    'window_height': 800,
    'video_display_size': (640, 480),
    'theme': 'default',  # 'default', 'dark', 'light'
    'font_family': 'Arial',
    'font_size': 12,
    'refresh_rate': 30,  # GUI refresh rate (ms)
}

# File Paths
PATHS = {
    'model_file': 'hand_sign_model.h5',
    'encoder_file': 'label_encoder.pkl',
    'custom_signs_file': 'custom_signs.json',
    'training_data_file': 'training_data.json',
    'config_file': 'config.py',
    'logs_dir': 'logs',
    'exports_dir': 'exports',
}

# ASL Basic Signs (A-Z, 0-9)
ASL_SIGNS = {
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

# Add numbers 0-9
for i in range(10):
    ASL_SIGNS[str(i)] = f'number {i} sign'

# Performance Settings
PERFORMANCE_SETTINGS = {
    'use_gpu': True,           # Use GPU if available
    'threading': True,         # Use threading for video processing
    'optimize_memory': True,   # Optimize memory usage
    'cache_predictions': True, # Cache predictions for better performance
}

# Logging Settings
LOGGING_SETTINGS = {
    'level': 'INFO',           # DEBUG, INFO, WARNING, ERROR
    'file_logging': True,
    'console_logging': True,
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# Export Settings
EXPORT_SETTINGS = {
    'default_format': 'txt',   # txt, csv, json
    'include_timestamps': True,
    'include_confidence': True,
    'auto_export': False,
    'export_interval': 60,     # Auto-export interval (seconds)
}

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        PATHS['logs_dir'],
        PATHS['exports_dir']
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def get_config_value(category, key, default=None):
    """Get a configuration value safely"""
    config_categories = {
        'video': VIDEO_SETTINGS,
        'mediapipe': MEDIAPIPE_SETTINGS,
        'model': MODEL_SETTINGS,
        'training': TRAINING_SETTINGS,
        'recognition': RECOGNITION_SETTINGS,
        'gui': GUI_SETTINGS,
        'performance': PERFORMANCE_SETTINGS,
        'logging': LOGGING_SETTINGS,
        'export': EXPORT_SETTINGS
    }
    
    if category in config_categories and key in config_categories[category]:
        return config_categories[category][key]
    
    return default

def set_config_value(category, key, value):
    """Set a configuration value"""
    config_categories = {
        'video': VIDEO_SETTINGS,
        'mediapipe': MEDIAPIPE_SETTINGS,
        'model': MODEL_SETTINGS,
        'training': TRAINING_SETTINGS,
        'recognition': RECOGNITION_SETTINGS,
        'gui': GUI_SETTINGS,
        'performance': PERFORMANCE_SETTINGS,
        'logging': LOGGING_SETTINGS,
        'export': EXPORT_SETTINGS
    }
    
    if category in config_categories and key in config_categories[category]:
        config_categories[category][key] = value
        return True
    
    return False

# Initialize directories
create_directories() 