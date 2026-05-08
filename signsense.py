

import sys
import time
import threading
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
def check_deps():
    missing = []
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python-headless")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")

    if missing:
        print("❌ Missing packages. Run:")
        for p in missing:
            print(f"   pip install {p} --pre")
        sys.exit(1)

check_deps()

import cv2
import numpy as np
from PIL import Image, ImageTk


# ── Hand Analyzer (pure OpenCV, no mediapipe) ─────────────────────────────────
class HandAnalyzer:
    def get_skin_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0,  20, 70],  dtype=np.uint8)
        upper1 = np.array([20, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 20, 70],  dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2)
        )
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.dilate(mask, k, iterations=2)
        return mask

    def count_fingers(self, contour, shape):
        if contour is None or len(contour) < 5:
            return 0, None
        hull_idx = cv2.convexHull(contour, returnPoints=False)
        hull_pts = cv2.convexHull(contour)
        if hull_idx is None or len(hull_idx) < 3:
            return 0, hull_pts
        try:
            defects = cv2.convexityDefects(contour, hull_idx)
        except cv2.error:
            return 0, hull_pts
        if defects is None:
            return 0, hull_pts

        count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d / 256.0 < 20:
                continue
            start = np.array(contour[s][0])
            end   = np.array(contour[e][0])
            far   = np.array(contour[f][0])
            a = np.linalg.norm(start - far)
            b = np.linalg.norm(end   - far)
            c = np.linalg.norm(start - end)
            if a * b == 0:
                continue
            angle = np.degrees(np.arccos(np.clip((a**2+b**2-c**2)/(2*a*b), -1, 1)))
            if angle < 90:
                count += 1
        return min(count + 1, 5), hull_pts

    def classify(self, fingers, contour):
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / (h + 1e-5)
        gestures = {
            0: ("Fist 👊",        "#E05C5C"),
            1: ("Pointing ☝️" if aspect < 0.5 else "Thumbs Up 👍", "#F5A623"),
            2: ("Peace ✌️",       "#50E3C2"),
            3: ("Three 🤟",       "#B57BFF"),
            4: ("Four 🖖",        "#FF8C42"),
            5: ("Hello / Wave 👋" if aspect > 0.85 else "Open Palm 🖐️", "#5CE05C"),
        }
        return gestures.get(fingers, ("Detecting...", "#888888"))

    def analyze(self, frame):
        mask = self.get_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, None, None, mask, "No hand detected 🤚", "#666666"
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 5000:
            return 0, None, None, mask, "Move hand closer ✋", "#666666"
        fingers, hull = self.count_fingers(contour, frame.shape)
        gesture, color = self.classify(fingers, contour)
        return fingers, contour, hull, mask, gesture, color


# ── Tkinter UI App ────────────────────────────────────────────────────────────
class SignSenseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✋ SignSense — Hand Sign Recognizer")
        self.root.configure(bg="#0D0D0D")
        self.root.resizable(False, False)

        self.analyzer     = HandAnalyzer()
        self.cap          = None
        self.running      = False
        self.show_mask    = False
        self.gesture_hist = []
        self.SMOOTH_N     = 6
        self.prev_time    = time.time()

        self._build_ui()
        self._start_camera()

    def _build_ui(self):
        # ── Title bar ──
        title_frame = tk.Frame(self.root, bg="#111111", pady=8)
        title_frame.pack(fill="x")

        tk.Label(title_frame, text="✋  S I G N S E N S E",
                 bg="#111111", fg="#00FFB2",
                 font=("Courier New", 16, "bold")).pack(side="left", padx=20)

        self.fps_label = tk.Label(title_frame, text="FPS: --",
                                   bg="#111111", fg="#555555",
                                   font=("Courier New", 11))
        self.fps_label.pack(side="right", padx=20)

        # ── Camera feed ──
        self.canvas = tk.Canvas(self.root, width=800, height=480,
                                 bg="#0D0D0D", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(5, 0))

        # ── Gesture display ──
        gesture_frame = tk.Frame(self.root, bg="#111827", pady=14)
        gesture_frame.pack(fill="x", padx=10, pady=6)

        tk.Label(gesture_frame, text="DETECTED SIGN",
                 bg="#111827", fg="#444444",
                 font=("Courier New", 9)).pack()

        self.gesture_label = tk.Label(gesture_frame,
                                       text="Show your hand 🤚",
                                       bg="#111827", fg="#00FFB2",
                                       font=("Courier New", 22, "bold"))
        self.gesture_label.pack()

        # ── Finger dots ──
        dots_frame = tk.Frame(self.root, bg="#0D0D0D", pady=6)
        dots_frame.pack()
        self.dot_canvases = []
        for i in range(5):
            c = tk.Canvas(dots_frame, width=28, height=28,
                          bg="#0D0D0D", highlightthickness=0)
            c.pack(side="left", padx=4)
            c.create_oval(2, 2, 26, 26, fill="#1A1A1A", outline="#333333", width=2, tags="dot")
            self.dot_canvases.append(c)

        # ── Controls ──
        ctrl_frame = tk.Frame(self.root, bg="#0D0D0D", pady=4)
        ctrl_frame.pack()

        btn_style = dict(bg="#1A1A2E", fg="#AAAAAA",
                         font=("Courier New", 10),
                         relief="flat", padx=14, pady=6,
                         cursor="hand2", activebackground="#2A2A4E",
                         activeforeground="#00FFB2")

        tk.Button(ctrl_frame, text="[ M ] Toggle Mask",
                  command=self._toggle_mask, **btn_style).pack(side="left", padx=6)
        tk.Button(ctrl_frame, text="[ Q ] Quit",
                  command=self._quit, **btn_style).pack(side="left", padx=6)

        tk.Label(self.root, text="SignSense v1.0  •  Python 3.14 Edition",
                 bg="#0D0D0D", fg="#2A2A2A",
                 font=("Courier New", 8)).pack(pady=(2, 8))

        # Key bindings
        self.root.bind("<q>", lambda e: self._quit())
        self.root.bind("<Q>", lambda e: self._quit())
        self.root.bind("<m>", lambda e: self._toggle_mask())
        self.root.bind("<M>", lambda e: self._toggle_mask())

    def _toggle_mask(self):
        self.show_mask = not self.show_mask

    def _quit(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.gesture_label.config(text="❌ No camera found!", fg="#FF5555")
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self._update_frame()

    def _update_dots(self, fingers, color):
        for i, c in enumerate(self.dot_canvases):
            c.delete("dot")
            if i < fingers:
                c.create_oval(2, 2, 26, 26, fill=color,
                              outline=color, width=2, tags="dot")
            else:
                c.create_oval(2, 2, 26, 26, fill="#1A1A1A",
                              outline="#333333", width=2, tags="dot")

    def _update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        fingers, contour, hull, mask, gesture, color = self.analyzer.analyze(frame)

        # Smooth gesture
        self.gesture_hist.append(gesture)
        if len(self.gesture_hist) > self.SMOOTH_N:
            self.gesture_hist.pop(0)
        smooth = max(set(self.gesture_hist), key=self.gesture_hist.count)

        # Draw contours
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 150), 2)
        if hull is not None:
            cv2.drawContours(frame, [hull], -1, (0, 200, 255), 2)

        # FPS
        now = time.time()
        fps = 1.0 / (now - self.prev_time + 1e-9)
        self.prev_time = now

        # Show mask or normal frame
        if self.show_mask:
            display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            display = frame

        # Convert to Tkinter image
        rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(rgb)
        imgtk   = ImageTk.PhotoImage(image=img)

        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)

        # Update UI labels
        self.fps_label.config(text=f"FPS: {fps:.0f}")
        self.gesture_label.config(text=smooth, fg=color)
        self._update_dots(fingers, color)

        self.root.after(15, self._update_frame)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  ✋  SignSense — Python 3.14 Edition")
    print("=" * 50)
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  OpenCV : {cv2.__version__}")
    print()
    print("  Controls:  M = toggle mask  |  Q = quit")
    print("=" * 50)

    root = tk.Tk()
    app  = SignSenseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()