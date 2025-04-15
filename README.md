# Voice Emotion Detector 🎙️

This project uses the RAVDESS dataset to detect emotions from voice recordings using Python and machine learning (Random Forest).

## Project Setup

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Place the RAVDESS dataset in `Dataset/RAVDESS/`.

## Model Training

To train the model, run:
```
python train_model.py
```

## Graphical User Interface (GUI)

You can run the emotion detector with a graphical interface:
```
python emotion_detector_gui.py
```

With the GUI you can:
- Record your voice for 5 seconds
- Analyze the emotion from the recording
- Get the results

## Command Line Interface

You can also detect emotions from the command line:
```
python predict_emotion.py
```

## Project Structure

- `train_model.py`: Trains the model using the RAVDESS dataset
- `extract_features.py`: Extracts MFCC features from audio files
- `predict_emotion.py`: Detects emotions from the command line
- `emotion_detector_gui.py`: Graphical user interface
- `Models/`: Trained model files
- `Dataset/RAVDESS/`: RAVDESS dataset

## Supported Emotions

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

## Requirements

For tkinter, it typically comes pre-installed with Python on Windows. On Linux, you can install it with:
```
sudo apt-get install python3-tk
``` 