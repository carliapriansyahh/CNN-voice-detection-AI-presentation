import os
import time
import numpy as np
import librosa
import cv2
from tensorflow.keras.models import load_model

# Path ke model CNN
MODEL_PATH = "/Users/carliapriansyahh/Downloads/pythonProject/voice_command_model.h5"
model = load_model(MODEL_PATH)

# Parameter audio
SAMPLE_RATE = 16000
DURATION = 1  # dalam detik
THRESHOLD = 0.8  # Confidence threshold untuk validasi prediksi

# Path folder PowerPoint slides
folderPath = "/Users/carliapriansyahh/Downloads/pythonProject/PowerPoint"
pathImages = sorted(os.listdir(folderPath))
imgNumber = 0

# Setup OpenCV untuk menampilkan slide
width, height = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)


# Fungsi untuk memuat audio dan memproses ke spectrogram
def predict_audio(audio_data):
    try:
        # Konversi audio ke mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        mel_spectrogram = mel_spectrogram[..., np.newaxis]
        mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)

        # Prediksi menggunakan model
        predictions = model.predict(mel_spectrogram)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        return predicted_class, confidence
    except Exception as e:
        print(f"Error predicting command: {e}")
        return None, 0


# Fungsi untuk merekam audio
def record_audio():
    import sounddevice as sd
    print("Listening...")
    audio_data = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    audio_data = audio_data.flatten()
    return audio_data


# Loop utama untuk menampilkan slide dan mendeteksi suara
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if imgNumber < len(pathImages):
        pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
        imgCurrent = cv2.imread(pathFullImage)

        if imgCurrent is None:
            print(f"Failed to load image at {pathFullImage}. Skipping to next image.")
            if imgNumber < len(pathImages) - 1:
                imgNumber += 1
            continue
    else:
        print("No more images in folder.")
        break

    # Rekam audio
    audio_data = record_audio()

    # Prediksi suara
    predicted_class, confidence = predict_audio(audio_data)

    if confidence > THRESHOLD:
        if predicted_class == 1:  # Right command
            print("Command: RIGHT (Next Slide)")
            if imgNumber < len(pathImages) - 1:
                imgNumber += 1
        elif predicted_class == 0:  # Left command
            print("Command: LEFT (Previous Slide)")
            if imgNumber > 0:
                imgNumber -= 1
    else:
        print("Command not clear, ignoring...")

    # Tampilkan slide
    cv2.imshow("Slides", imgCurrent)

    # Berikan opsi untuk keluar
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()