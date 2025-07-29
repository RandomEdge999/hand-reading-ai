# Hand Sign Recognition System

A comprehensive AI-powered hand sign recognition system that uses your webcam to interpret hand gestures and convert them to text. The system supports basic American Sign Language (ASL), custom hand symbols, and machine learning capabilities for continuous improvement.

## Features

### 🎯 Core Features
- **Real-time hand sign recognition** using webcam input
- **Text output display** below the video feed
- **Basic ASL support** for letters A-Z and numbers 0-9
- **Custom hand symbol learning** - teach the system new signs
- **Local AI processing** - runs entirely on your laptop
- **Modern GUI interface** with intuitive controls

### 🧠 AI & Learning Capabilities
- **Neural network-based recognition** using TensorFlow
- **MediaPipe hand tracking** for precise landmark detection
- **Training data collection** for custom signs
- **Model persistence** - saves learned patterns
- **Continuous learning** from user interactions

### 🎨 User Interface
- **Beautiful GUI** built with tkinter
- **Real-time video display** with hand landmark visualization
- **Text recognition history** tracking
- **Export functionality** for recognized text
- **Training interface** for adding new signs

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Windows 10/11 (tested on Windows 10)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd hand_reading_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import cv2, mediapipe, tensorflow; print('All dependencies installed successfully!')"
   ```

## Usage

### Quick Start

1. **Run the main application**
   ```bash
   python hand_sign_recognition.py
   ```

2. **Use the GUI version (recommended)**
   ```bash
   python gui_app.py
   ```

### Training Custom Signs

1. **Start the training system**
   ```bash
   python train_model.py
   ```

2. **Collect training data**
   - Use command: `collect <sign_name>`
   - Hold your hand in the desired position
   - Press 's' to save samples (collect 50 samples per sign)
   - Press 'q' to quit collection

3. **Manage your signs**
   - `list` - View all trained signs
   - `delete <sign_name>` - Remove a sign
   - `quit` - Exit training

### GUI Controls

- **Clear Text** - Clear the recognized text
- **Save Custom Sign** - Save current hand position as a new sign
- **Train Model** - Open training interface
- **View History** - See recognition history
- **Export Text** - Save recognized text to file

## System Architecture

### Components

1. **Hand Detection** (`mediapipe`)
   - Real-time hand landmark detection
   - 21 3D landmarks per hand
   - Robust tracking with confidence scores

2. **Feature Extraction**
   - Converts landmarks to feature vectors
   - Normalized coordinates (x, y, z)
   - 63-dimensional feature space

3. **Neural Network Model**
   - Input: 21 landmarks × 3 coordinates = 63 features
   - Hidden layers: 128 → 64 neurons
   - Output: 36 classes (26 letters + 10 numbers)
   - Activation: ReLU with dropout for regularization

4. **Learning System**
   - Custom sign storage
   - Training data collection
   - Model retraining capabilities

### File Structure

```
hand_reading_AI/
├── hand_sign_recognition.py    # Main recognition application
├── gui_app.py                  # GUI version
├── train_model.py              # Training system
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── hand_sign_model.h5          # Trained model (created after first run)
├── label_encoder.pkl           # Label mappings (created after first run)
├── custom_signs.json           # Custom signs database
└── training_data.json          # Training data collection
```

## Customization

### Adding New Signs

1. **Through GUI**
   - Position your hand in the desired gesture
   - Click "Save Custom Sign"
   - Enter a name for the sign

2. **Through Training System**
   - Run `python train_model.py`
   - Use `collect <sign_name>` command
   - Collect multiple samples for better accuracy

### Model Training

The system uses a simple neural network that can be enhanced:

1. **Architecture modifications** - Edit `create_initial_model()` in `hand_sign_recognition.py`
2. **Feature engineering** - Modify `extract_hand_features()` for better features
3. **Data augmentation** - Add rotation, scaling, and noise to training data

### Performance Optimization

- **Reduce model complexity** for faster inference
- **Adjust MediaPipe confidence thresholds** for better detection
- **Use GPU acceleration** with TensorFlow-GPU
- **Optimize video resolution** for your hardware

## Troubleshooting

### Common Issues

1. **Webcam not detected**
   - Check webcam permissions
   - Try different camera index (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)
   - Ensure no other application is using the camera
   - Install or update webcam drivers
   - On Windows/macOS, verify camera privacy settings
   - If the application shows a "Camera Error" message, review these hints

2. **Poor recognition accuracy**
   - Ensure good lighting conditions
   - Keep hand clearly visible in frame
   - Collect more training samples
   - Adjust hand position to match ASL standards

3. **Performance issues**
   - Reduce video resolution
   - Close other applications
   - Use lower MediaPipe confidence thresholds
   - Consider using a lighter model architecture

4. **Dependencies installation errors**
   - Update pip: `python -m pip install --upgrade pip`
   - Install Visual C++ build tools (Windows)
   - Use conda instead of pip for problematic packages

### System Requirements

- **Minimum**: 4GB RAM, Intel i3 or equivalent
- **Recommended**: 8GB RAM, Intel i5 or better
- **GPU**: Optional but recommended for better performance
- **Storage**: 2GB free space for models and data

## Future Enhancements

### Planned Features
- **Multi-hand support** for complex gestures
- **Dynamic gesture recognition** (moving hands)
- **Voice output** for recognized signs
- **Mobile app version**
- **Cloud-based model sharing**
- **Advanced ASL grammar support**

### Technical Improvements
- **Transformer-based models** for better accuracy
- **Real-time model fine-tuning**
- **Cross-platform compatibility**
- **Web-based interface**
- **API for third-party integration**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Note**: This system is designed for educational and accessibility purposes. For professional ASL interpretation, please consult certified interpreters and appropriate resources. 