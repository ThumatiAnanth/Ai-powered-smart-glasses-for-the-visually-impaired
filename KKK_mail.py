

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders



#this code can do text, image, faces and can stop through voice

import cv2
import numpy as np
import pickle
import pyttsx3
import speech_recognition as sr
import datetime
import google.generativeai as genai
import os
import threading
from PIL import Image
import pytesseract

# Initialize text-to-speech engine
engine = pyttsx3.init()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def speak(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

# Function to recognize voice commands
def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"Recognized: {command}")
        return command
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return None

# Function to get current time and date
def get_time_date():
    now = datetime.datetime.now()
    time = now.strftime("%I:%M %p")
    date = now.strftime("%B %d, %Y")
    return time, date

# Function to interact with Gemini-Pro model
def chat_with_gemini(query, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    prompt = f"Give me the output in 2 to 3 lines for this question, {query}"
    response = model.generate_content(prompt)
    speak(response.text)

# Function to capture and describe an image using Gemini-1.5 Flash model
def describe_image(api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Capture an image from the live camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        speak("Failed to capture image.")
        return

    # Convert OpenCV image to PIL format
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    # Generate response using Gemini
    response = model.generate_content(["Describe the following image in 2-3 lines:", pil_image])

    # Speak the description
    speak(response.text)

# Global flag to stop face recognition
stop_flag = False

# Function to listen for the "stop" command
def listen_for_stop():
    global stop_flag
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        while not stop_flag:
            try:
                print("Listening for 'stop' command...")
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio).lower()
                if "stop" in command:
                    print("Stop command received. Exiting face recognition...")
                    stop_flag = True  # Set flag to True to stop face recognition
                    break
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                print("Error connecting to speech recognition service.")


def extract_text_from_camera():
    """Extract text from an image using OCR."""
    global stop_flag
    stop_flag = False  # Reset flag before starting
    listener_thread = threading.Thread(target=listen_for_stop, daemon=True)
    listener_thread.start()

    cap = cv2.VideoCapture(1)

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Text Detection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            message = "Detected Text: " + text.strip()
            print(message)
            speak(message)

    cap.release()
    cv2.destroyAllWindows()
    print("Text extraction stopped.")
# Function for face recognition
def recognize_faces():
    """Recognizes faces from the live camera feed and stops when 'stop' is said."""
    global stop_flag
    stop_flag = False  # Reset flag before starting

    # Load trained face recognizer model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read("face_model.xml")

    # Load label map
    with open("labels.pkl", "rb") as f:
        label_map = pickle.load(f)

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Start voice command listener in a separate thread
    listener_thread = threading.Thread(target=listen_for_stop, daemon=True)
    listener_thread.start()

    cap = cv2.VideoCapture(1)

    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            label_id, confidence = face_recognizer.predict(face_img)

            # Adjust confidence threshold
            if confidence < 55:  # Lower value means better match
                name = label_map.get(label_id, "Unknown")
            else:
                name = "Unknown"

            confidence_text = f"{name} ({round(100 - confidence, 2)}%)"
            cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Speak the recognized name
            if name != "Unknown":
                speak(f"{name} is recognized")
            else:
                speak("Unknown person is recognized")

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == ord("q"):  # Manual exit option
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Face recognition stopped.")


def send_alert_email():
    sender_email = 'lifeandliving1304@gmail.com'
    sender_password = 'Pranay@22'
    receiver_email = 'pranaythumati@gmail.com'

    subject = "Emergency Alert: Help Command Received"
    body = "An emergency 'help' command has been triggered. Please take immediate action."

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, 'rqop cthw egjx mkrg')
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        speak("Alert email sent successfully.")
        print("Alert email sent successfully.")
    except Exception as e:
        speak("Failed to send the alert email.")
        print(f"Error: {e}")



# Main function
def main():
    api_key = "AIzaSyA55ASfpfA1nHaShOwIKOezkNhThl9GFGY"

    while True:
        command = listen_command()
        if command:
            if "time" in command:
                time, _ = get_time_date()
                speak(f"The time is {time}")
            elif "date" in command:
                _, date = get_time_date()
                speak(f"Today's date is {date}")
            elif "gemini" in command:
                speak("What would you like to ask?")
                query = listen_command()
                if query:
                    chat_with_gemini(query, api_key)
            elif "describe image" in command or "describe" in command:
                describe_image(api_key)
            elif "recognize faces" in command or "detect" in command:
                recognize_faces()
            elif "hello" in command or "text" in command:
                extract_text_from_camera()
            elif "help" in command or "toy" in command:
                speak("Sending an alert email now.")
                send_alert_email()
            elif "exit" in command:
                speak("Goodbye!")
                break

if __name__ == "__main__":
    main()