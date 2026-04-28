import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import transforms
from fer2013 import EmotionCNN

#Settings the list of emotions
EMOTIONS = ["angr", "disg", "fear", "happ", "neut", "sadn", "surp"]

EMOTION_COLORS = {
    "angry":    (170,   0,   0),
    "disgust":  (  0, 170,   0),
    "fear":     (  0, 170, 170),
    "happy":    (170, 170,   0),
    "neutral":  (170, 170, 170),
    "sad":      (  0,   0, 170),
    "surprise": (170,   0, 170),
}

#loading the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN()
models_dir = "models"
if os.path.exists("best.pth"):
    model_path = "best.pth"
else:
    if os.path.isdir(models_dir) and os.listdir(models_dir):
        model_path = os.path.join(models_dir, sorted(os.listdir(models_dir))[-1])
    else:
        print("No models found in 'models' directory.")
print(f"Loading {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

cascade_path = os.path.join("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

transform = transforms.Compose([
    transforms.Grayscale(),          # ensure 1 channel
    transforms.Resize((48, 48)),     # FER standard size
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def infer_emotion(face):
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs

class App:
    def __init__(self, window,):
        self.window = window
        self.window.title("Emotion Recognition")

        self.video_source = 0 # default webcam
        self.vid = cv2.VideoCapture(self.video_source) #Open the webcam

        self.video_label = tk.Label(root) #Label to display the video feed
        self.video_label.pack() #Pack the label into the window

        self.frame_count = 0
        self.last_boxes = []
        self.last_text = []

        btn_frame = tk.Frame(window) #Frame to hold the buttons
        btn_frame.pack() #Pack the frame into the window

        #Adding buttons to pause and quit the webcam feed
        tk.Button(btn_frame, text="Pause", width=10, command=self.toggle).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Quit" , width=10, command=self.quit  ).pack(side=tk.LEFT, padx=5)
        
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
                face_coords = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48))
                if self.frame_count >= 15:
                    if len(face_coords) > 0:
                        # self.frame_count = 0
                        self.last_boxes = []
                        self.last_text = []
                        for x, y, w, h in face_coords:
                            face = frame[y:y + h, x:x + w]
                            model_guess = list(zip(EMOTIONS, infer_emotion(face)))
                            # top3 = sorted(model_guess, key=lambda pair: pair[1], reverse=True)[:3]
                            # guess_string = ' | '.join([f'{emo[0]} {round(emo[1]*100)}%' for emo in top3])
                            guess_string = [(emo[0], "."*round(emo[1]*20)) for emo in model_guess]
                            
                            # draw rectangle around face
                            rect_args = ((x, y), (x + w, y + h))
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            self.last_boxes.append(rect_args)
                            # print guess_string above rectangle
                            # cv2.putText(frame, guess_string, (x+w+5, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            next_last_text = []
                            for i in range(len(guess_string)):
                                for j in range(2):
                                    text_args = (guess_string[i][j], (x+w+5+(40*j), y+10+(20*i)))
                                    cv2.putText(frame, guess_string[i][j], (x+w+5+(40*j), y+10+(20*i)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                                    next_last_text.append(text_args)
                            self.last_text.append(next_last_text)
                else:
                    self.frame_count += 1
                    for i in range(len(self.last_boxes)):
                        cv2.rectangle(frame, self.last_boxes[i][0], self.last_boxes[i][1], (0, 255, 0), 2)
                        for j in range(len(self.last_text[i])):
                            cv2.putText(frame, self.last_text[i][j][0], self.last_text[i][j][1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_label.imgtk = img
                self.video_label.configure(image=img)
        self.window.after(20, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()