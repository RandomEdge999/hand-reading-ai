# Advanced Hand Sign Recognition System

A powerful, local AI-powered hand sign recognition system that uses your webcam to interpret hand gestures and convert them to text. The system runs entirely on your local machine using advanced machine learning models and MediaPipe's robust hand tracking.

## 🚀 Key Features

### 🎯 Core Capabilities
- **Real-time hand sign recognition** using webcam input
- **Local AI processing** - runs entirely on your laptop, no internet required
- **Advanced MediaPipe hand tracking** with 21 3D landmarks
- **XGBoost + Random Forest ensemble models** for superior accuracy
- **High-resolution camera support** (1280x720)
- **Confidence scoring** for reliable predictions

### 🧠 Advanced AI & Learning
- **Ensemble machine learning** using XGBoost and Random Forest
- **Advanced feature extraction** including finger lengths, angles, and hand geometry
- **Prediction smoothing** with confidence-based filtering
- **Custom sign learning** - teach the system new gestures
- **Model persistence** - saves learned patterns locally
- **Real-time model training** capabilities

### 🎨 Modern User Interface
- **Beautiful dark-themed GUI** built with tkinter
- **Real-time video display** with hand landmark visualization
- **Live confidence display** showing prediction reliability
- **Text recognition history** with timestamps
- **Export functionality** for recognized text
- **Custom sign management** interface

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Windows 10/11 (tested on Windows 10)
- 4GB+ RAM recommended

### Quick Setup

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd hand-reading-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the system**
   ```bash
   python test_advanced_system.py
   ```

## 🚀 Usage

### Quick Start

1. **Launch the advanced system**
   ```bash
   python advanced_launcher.py
   ```

2. **Or run the GUI directly**
   ```bash
   python advanced_gui_app.py
   ```

3. **For command line version**
   ```bash
   python advanced_hand_recognition.py
   ```

### System Controls

- **Clear Text** - Clear the recognized text
- **Save Custom Sign** - Save current hand position as a new sign
- **Train Model** - Retrain models with collected data
- **View History** - See recognition history with confidence scores
- **Export Text** - Save recognized text to file
- **Settings** - Adjust confidence threshold and other parameters

## 🏗️ System Architecture

### Advanced Components

1. **MediaPipe Hand Detection**
   - 21 3D landmarks per hand
   - High confidence thresholds (0.8 detection, 0.7 tracking)
   - Support for multiple hands
   - Complex model for better accuracy

2. **Advanced Feature Extraction**
   - Basic landmark coordinates (63 features)
   - Hand size metrics (width, height, depth)
   - Finger length calculations
   - Palm center coordinates
   - Inter-finger angles
   - Total: 100+ advanced features

3. **Ensemble Machine Learning**
   - **XGBoost Classifier**: 200 estimators, optimized parameters
   - **Random Forest**: 150 estimators, robust backup model
   - **Ensemble Decision**: Uses highest confidence prediction
   - **Feature Scaling**: StandardScaler for normalization

4. **Prediction Pipeline**
   - Real-time feature extraction
   - Model ensemble prediction
   - Confidence-based filtering
   - Prediction smoothing buffer
   - Text accumulation with history

### File Structure

```
hand-reading-ai/
├── advanced_hand_recognition.py    # Main recognition engine
├── advanced_gui_app.py             # Advanced GUI application
├── advanced_launcher.py            # System launcher
├── test_advanced_system.py         # Comprehensive testing
├── requirements.txt                # Dependencies
├── config.py                       # Configuration settings
├── README.md                       # This file
├── advanced_xgb_model.pkl          # XGBoost model (created after first run)
├── advanced_rf_model.pkl           # Random Forest model
├── advanced_scaler.pkl             # Feature scaler
├── advanced_label_encoder.pkl      # Label mappings
├── advanced_custom_signs.json      # Custom signs database
└── training_data.json              # Training data collection
```

## 🎯 Supported Signs

### ASL Alphabet (A-Z)
- **A**: Thumb out, fingers closed
- **B**: All fingers extended, thumb tucked
- **C**: Curved hand like letter C
- **D**: Index finger up, others closed
- **E**: All fingers closed, thumb across palm
- **F**: Index and thumb touching, other fingers up
- **G**: Index finger pointing
- **H**: Index and middle finger extended
- **I**: Pinky finger up, others closed
- **J**: Pinky finger moving in J motion
- **K**: Index and middle finger up, thumb between
- **L**: Thumb and index finger forming L
- **M**: Three fingers down, thumb tucked
- **N**: Two fingers down, thumb tucked
- **O**: All fingers curved together
- **P**: Index finger pointing down
- **Q**: Index finger pointing down, thumb out
- **R**: Index and middle finger crossed
- **S**: Fist with thumb over fingers
- **T**: Index finger up, thumb touching middle
- **U**: Index and middle finger together up
- **V**: Index and middle finger apart up
- **W**: Three fingers up (thumb, index, middle)
- **X**: Index finger bent
- **Y**: Thumb and pinky out
- **Z**: Index finger moving in Z motion

### Numbers (0-9)
- All standard ASL number signs

### Special Signs
- **SPACE**: Open palm facing forward
- **DELETE**: Closed fist
- **ENTER**: Thumbs up

## ⚙️ Configuration

### Performance Settings
- **Confidence Threshold**: 0.85 (adjustable in GUI)
- **Prediction Delay**: 1.0 seconds between predictions
- **Buffer Size**: 5 predictions for smoothing
- **Camera Resolution**: 1280x720 (configurable)

### Model Settings
- **XGBoost**: 200 estimators, max depth 8
- **Random Forest**: 150 estimators, max depth 10
- **Feature Scaling**: StandardScaler normalization
- **Ensemble Weighting**: Confidence-based selection

## 🔧 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check webcam permissions in Windows settings
   - Try different camera index (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)
   - Ensure no other application is using the camera
   - Update webcam drivers

2. **Poor recognition accuracy**
   - Ensure good lighting conditions
   - Keep hand clearly visible in frame
   - Adjust confidence threshold in settings
   - Train custom signs for better accuracy

3. **Performance issues**
   - Reduce camera resolution in config
   - Close other applications
   - Lower confidence threshold
   - Use command line version for better performance

4. **Dependencies installation errors**
   - Update pip: `python -m pip install --upgrade pip`
   - Install Visual C++ build tools (Windows)
   - Use conda instead of pip for problematic packages

### System Requirements

- **Minimum**: 4GB RAM, Intel i3 or equivalent
- **Recommended**: 8GB RAM, Intel i5 or better
- **GPU**: Optional but recommended for better performance
- **Storage**: 2GB free space for models and data

## 🚀 Advanced Features

### Custom Sign Training
1. Show your hand in the desired position
2. Click "Save Custom Sign" in the GUI
3. Enter a name for the sign
4. The system will learn and recognize this sign

### Model Training
- Collect multiple samples of each sign
- Use the "Train Model" button to retrain
- Models automatically save and improve over time

### Export and History
- Export recognized text to files
- View detailed recognition history
- Track confidence scores over time

## 🔮 Future Enhancements

### Planned Features
- **Multi-hand support** for complex gestures
- **Dynamic gesture recognition** (moving hands)
- **Voice output** for recognized signs
- **Mobile app version**
- **Cloud model sharing** (optional)
- **Advanced ASL grammar support**

### Technical Improvements
- **Transformer-based models** for better accuracy
- **Real-time model fine-tuning**
- **Cross-platform compatibility**
- **Web-based interface**
- **API for third-party integration**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues, questions, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation
- Run `python test_advanced_system.py` for diagnostics

---

**Note**: This system is designed for educational and accessibility purposes. For professional ASL interpretation, please consult certified interpreters and appropriate resources.

## 🎉 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Test the system
python test_advanced_system.py

# Launch the system
python advanced_launcher.py

# Or run GUI directly
python advanced_gui_app.py
``` 