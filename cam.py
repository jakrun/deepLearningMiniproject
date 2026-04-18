import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import transforms
from fer2013 import EmotionCNN

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionCNN()
models_dir = "models"
if os.path.isdir(models_dir) and os.listdir(models_dir):
    model_path = os.path.join(models_dir, sorted(os.listdir(models_dir))[-1])
else:
    model_path = "best.pth"
print(f"Loading {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def predict(face_bgr):
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


def draw_emotion_wheel(frame, center, radius, top_emotions):
    """top_emotions: list of (name, prob), already sorted descending. Draws a donut where the 3 wedges span their actual probabilities (remainder shown as dark ring)."""
    cx, cy = center
    thickness = max(10, radius // 2)

    cv2.ellipse(frame, (cx, cy), (radius, radius), 0, 0, 360, (40, 40, 40), thickness, cv2.LINE_AA)

    start = -90.0
    for name, prob in top_emotions:
        sweep = float(prob) * 360.0
        color = EMOTION_COLORS.get(name, (255, 255, 255))
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start, start + sweep, color, thickness, cv2.LINE_AA)
        start += sweep

    top_prob = top_emotions[0][1]
    txt = f"{int(round(top_prob * 100))}%"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, txt, (cx - tw // 2, cy + th // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_legend(frame, top_left, top_emotions):
    x, y = top_left
    row_h = 20
    for i, (name, prob) in enumerate(top_emotions):
        color = EMOTION_COLORS.get(name, (255, 255, 255))
        ry = y + i * row_h
        cv2.rectangle(frame, (x, ry), (x + 14, ry + 14), color, -1)
        cv2.putText(frame, f"{name} {int(round(prob * 100))}%",
                    (x + 20, ry + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Cam")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam")

        self.video_label = tk.Label(root)
        self.video_label.pack(padx=10, pady=10)

        self.pred_var = tk.StringVar(value="Prediction: -")
        tk.Label(root, textvariable=self.pred_var, font=("Segoe UI", 16)).pack(pady=(0, 10))

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0, 10))

        self.running = True
        self.smoothed_probs = None
        self.smoothing_alpha = 0.15
        tk.Button(btn_frame, text="Pause", width=10, command=self.toggle).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Quit", width=10, command=self.quit).pack(side=tk.LEFT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.update_frame()

    def toggle(self):
        self.running = not self.running

    def update_frame(self):
        if self.running:
            ok, frame = self.cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    face = frame[y:y + h, x:x + w]
                    probs = predict(face)
                    if self.smoothed_probs is None:
                        self.smoothed_probs = probs.copy()
                    else:
                        a = self.smoothing_alpha
                        self.smoothed_probs = a * probs + (1 - a) * self.smoothed_probs
                    display_probs = self.smoothed_probs
                    order = sorted(range(len(display_probs)), key=lambda i: display_probs[i], reverse=True)
                    top3 = [(EMOTIONS[i], float(display_probs[i])) for i in order[:3]]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    radius = max(20, h // 8)
                    margin = 10
                    cx = x - margin - radius
                    cy = y + radius
                    if cx - radius < 0:
                        cx = x + w + margin + radius
                    draw_emotion_wheel(frame, (cx, cy), radius, top3)

                    legend_x = cx - radius
                    legend_y = cy + radius + 8
                    draw_legend(frame, (legend_x, legend_y), top3)

                    self.pred_var.set(
                        "  ".join(f"{n} {int(round(p * 100))}%" for n, p in top3)
                    )
                else:
                    self.pred_var.set("No face detected")

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(rgb))
                self.video_label.imgtk = img
                self.video_label.configure(image=img)

        self.root.after(20, self.update_frame)

    def quit(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
