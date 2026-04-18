import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import transforms
from fer2013 import EmotionCNN

#Settings the list of emotions
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_COLORS = {
    "angry":    (0, 0, 255),
    "disgust":  (0, 180, 0),
    "fear":     (200, 0, 200),
    "happy":    (0, 230, 255),
    "neutral":  (200, 200, 200),
    "sad":      (255, 80, 0),
    "surprise": (0, 165, 255),
}

#loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN()
models_dir = "models"
if os.path.isdir(models_dir) and os.listdir(models_dir):
    model_path = os.path.join(models_dir, sorted(os.listdir(models_dir))[-1])   
else:
    print("No models found in 'models' directory.")
print(f"Loading {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

class App:
    def __init__(self, window,):
        self.window = window
        self.window.title("Emotion Recognition")

        self.video_source = 0 # default webcam
        self.vid = cv2.VideoCapture(self.video_source) #Open the webcam

        self.video_label = tk.Label(root) #Label to display the video feed
        self.video_label.pack() #Pack the label into the window


        btn_frame = tk.Frame(window) #Frame to hold the buttons
        btn_frame.pack() #Pack the frame into the window

        #Adding buttons to pause and quit the webcam feed
        tk.Button(btn_frame, text="Pause", width=10, command=self.toggle).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Quit", width=10, command=self.quit).pack(side=tk.LEFT, padx=5)
        
        #Start the update loop to continuously get frames from the webcam.
        self.running = True
        self.update()

    def toggle(self): #So that we can pause the webcam to examine
        self.running = not self.running

    def quit(self): #To quit the webcam and close the window
        self.running = False
        if self.vid.isOpened():
            self.vid.release()
        self.window.destroy()
    
    def update(self):
        if self.running:
            ok, frame = self.vid.read()
            if ok:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
                #So we feed the grayscale image to the face detector, as it works better on single channel images
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
                if len(faces) > 0:
                    pass
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_label.imgtk = img
                self.video_label.configure(image=img)                
        self.window.after(20, self.update)        


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()