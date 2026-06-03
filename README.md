# SignSense — Real-time Sign Language Recognition

A real-time hand gesture recognition app that detects sign language gestures and speaks them aloud — built to help deaf and hard-of-hearing individuals communicate with hearing people using just a webcam.

## Demo

![SignSense Demo](demo.png)

## Problem
Deaf and hard-of-hearing individuals face communication barriers with people who don't know sign language. SignSense bridges that gap by detecting hand gestures in real time and converting them to voice output — no internet, no ML model, no special hardware required.

## How It Works
- Detects skin using HSV color range masking
- Finds hand contour using OpenCV
- Counts fingers using convexity defects algorithm
- Classifies gesture and speaks it aloud instantly

## Gestures Supported
| Gesture | Sign |
|---|---|
| Fist | 👊 |
| Pointing | ☝️ |
| Thumbs Up | 👍 |
| Peace | ✌️ |
| Three | 🤟 |
| Four | 🖖 |
| Hello / Wave | 👋 |
| Open Palm | 🖐️ |

## Tech Stack
- Python 3.14
- OpenCV
- Tkinter
- NumPy
- Pillow
- pyttsx3 (voice output)

## Performance
- Runs at 30+ FPS in real time
- Voice output speaks detected gesture aloud
- No internet connection required
- No ML model — pure OpenCV computer vision pipeline
- Works on any laptop with a webcam

## How to Run

```bash
pip install opencv-python-headless numpy pillow pyttsx3
python signsense.py
```

**Controls:**
- `M` — toggle mask view
- `Q` — quit

## Future Improvements
- Add full ISL (Indian Sign Language) alphabet support
- Mobile version using Flutter
- Two-way communication — speech to text for deaf users
