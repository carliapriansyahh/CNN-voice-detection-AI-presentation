
Voice Command Slide Control System

This project demonstrates a voice-controlled system for navigating PowerPoint slides using a CNN-based voice command model. The application allows users to control slide navigation (Next and Previous) using simple voice commands (“Right” for next, “Left” for previous). This implementation is designed for accessibility and can be useful in scenarios where hands-free control is preferred.

Files in This Project

1. PowerPoint/: Folder containing sample slides for testing the application.
2. voice_command_model.h5: Trained CNN model for recognizing voice commands (“Right” and “Left”).
3. Test Aplication.py: Main Python script to run the application, capturing audio input and controlling the slides based on recognized commands.
4. README.md: Documentation for setting up and running the application.

Prerequisites

1. Python 3.x
2. Required packages: TensorFlow, OpenCV, numpy, librosa, sounddevice

You can install the necessary packages using:

1. pip install tensorflow opencv-python numpy librosa sounddevice

Running the Application

1. Ensure the voice_command_model.h5 and PowerPoint/ folder are in the same directory as Test Aplication.py.
2.	Run the application using the command:

python Test\ Aplication.py

1. The application will start your webcam and listen for audio commands. Use the following commands to control the slides:
2. Say “Right” to move to the next slide.
3. Say “Left” to go to the previous slide.

Notes

  ##The model has been trained to recognize commands with a confidence threshold of 0.8 for accurate command detection.##
  ##The application is designed for MacBook microphones and should work with AirPods as an input device. Adjustments may be needed for other audio setups.##

