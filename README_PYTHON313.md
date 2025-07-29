# Hand Sign Recognition System - Python 3.13 Compatible

A simplified but functional hand sign recognition system that works with Python 3.13 and available packages.

## 🎯 Features

- **Real-time hand detection** using skin color analysis
- **Webcam input** with mirror effect
- **Custom sign learning** - save your own hand gestures
- **Visual feedback** - see hand contours and detection status
- **Python 3.13 compatible** - works with the latest Python version

## 📋 Requirements

- Python 3.13
- Webcam
- Windows 10/11

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python hand_sign_demo.py
```

### 3. Alternative: Use the Launcher

```bash
python run_basic_system.py
```

## 🎮 How to Use

1. **Start the system** - Run the demo script
2. **Show your hand** - Position your hand in front of the camera
3. **Watch detection** - Green contour shows detected hand
4. **Save custom signs** - Press 's' when hand is detected
5. **Clear text** - Press 'c' to clear the text display
6. **Quit** - Press 'q' to exit

## 📁 Files

- `hand_sign_demo.py` - Main demo application
- `run_basic_system.py` - GUI launcher
- `requirements.txt` - Compatible dependencies
- `custom_signs.json` - Saved custom signs (created automatically)

## 🔧 Controls

- **'q'** - Quit the application
- **'s'** - Save current hand position as custom sign
- **'c'** - Clear the text display

## 🎨 What You'll See

- **Main Window**: Live camera feed with hand detection overlay
- **Skin Detection Window**: Shows the skin color mask
- **Text Display**: Shows recognized text and custom signs
- **Detection Rate**: Percentage of frames where hand was detected

## 🧠 How It Works

1. **Skin Detection**: Uses HSV color space to detect skin-colored regions
2. **Contour Analysis**: Finds the largest skin-colored contour (your hand)
3. **Feature Extraction**: Converts hand region to numerical features
4. **Custom Sign Storage**: Saves hand features for later recognition

## 📊 Performance Tips

- **Good lighting**: Ensure your hand is well-lit
- **Clean background**: Avoid skin-colored objects in background
- **Hand positioning**: Keep your hand clearly visible to the camera
- **Distance**: Position hand 20-50cm from camera

## 🔮 Future Enhancements

This is a simplified version. For full functionality, consider:

1. **Downgrade Python**: Use Python 3.11 for MediaPipe support
2. **Use TensorFlow**: Install TensorFlow for advanced recognition
3. **Add Pre-trained Models**: Include ASL recognition models

## 🛠️ Troubleshooting

### Camera Not Working
- Check webcam permissions
- Try different camera indices (0, 1, -1)
- Ensure no other app is using the camera

### Poor Detection
- Improve lighting conditions
- Adjust hand position
- Check skin color range in code

### Import Errors
- Run: `pip install -r requirements.txt`
- Check Python version: `python --version`

## 📈 Detection Statistics

The system shows:
- **Frame count**: Total frames processed
- **Detection rate**: Percentage of frames with hand detected
- **Hand area**: Size of detected hand contour

## 🎯 Customization

You can modify the skin detection parameters in `hand_sign_demo.py`:

```python
# Adjust these values for different lighting/skin tones
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)
```

## 📝 Notes

- This is a **demo version** focused on hand detection
- **No pre-trained recognition** - saves custom signs only
- **Skin color based** - may need adjustment for different skin tones
- **Python 3.13 compatible** - uses only available packages

## 🎉 Success!

When working correctly, you should see:
- Green contour around your hand
- "Hand Detected!" message
- Detection rate above 50%
- Ability to save custom signs

---

**Enjoy exploring hand sign recognition!** 🤚✨ 