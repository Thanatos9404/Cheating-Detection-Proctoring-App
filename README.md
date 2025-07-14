# 👁️ Cheat Detector – Proctoring System using OpenCV & MediaPipe

A real-time webcam-based cheating detection system for **online exams** using Python, OpenCV, and MediaPipe. It monitors suspicious behavior like:
- Talking
- Face hiding or obstruction
- Looking away from the screen
- Multiple people in frame
- Sudden motion

All detected events are logged, screenshots and video clips are saved, and short audio recordings are captured for post-exam review.

---

## 📸 Features

- ✅ Real-time face detection and tracking  
- ✅ Head movement & gaze direction estimation  
- ✅ Talking detection based on lip movement  
- ✅ Motion detection using frame differencing  
- ✅ Audio recording triggered on suspicious events  
- ✅ Automatic saving of:
  - Screenshots (`/cheating_screenshots`)
  - Video clips (`/cheating_clips`)
  - Audio evidence (`/cheating_audio`)
  - Timestamps and reasons (`cheating_log.txt`)

---

## 🔧 Tech Stack

| Technology | Use |
|------------|-----|
| Python 3.x | Core language |
| OpenCV     | Video processing and motion detection |
| MediaPipe  | Face mesh and detection |
| NumPy      | Calculations |
| sounddevice & scipy.io.wavfile | Audio recording |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/cheat-detector.git
cd cheat-detector
````

### 2. Install dependencies

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install required packages:

```bash
pip install opencv-python mediapipe sounddevice scipy numpy
```

---

## ▶️ How to Run

Simply execute the script:

```bash
python cheat_detector.py
```

Press **`q`** to quit the proctoring window.

---

## 📂 Output Files

| Folder                  | Description                          |
| ----------------------- | ------------------------------------ |
| `/cheating_screenshots` | Snapshot of screen at time of event  |
| `/cheating_clips`       | 10-second video from time of event   |
| `/cheating_audio`       | Audio of the user during event       |
| `cheating_log.txt`      | Time-stamped list of detected events |

---

## 🚨 Detected Behaviors

* **Face not visible**: Covered or out of frame
* **Multiple faces**: Presence of more than one person
* **Significant motion**: Sudden camera/subject movement
* **Looking away**: Detected using head pose estimation
* **Talking detected**: Based on lip distance threshold

---

## 🧠 How It Works

* Face detection is done using **MediaPipe's Face Detection** & **Face Mesh**.
* Gaze/head direction is estimated using **6 landmark points** and `cv2.solvePnP`.
* Lip distance is used to infer speaking activity.
* Frame differencing detects sudden movements.

---

## 📌 Notes

* Ensure a well-lit environment for best accuracy.
* Frame rate is capped at \~20 FPS to avoid CPU overload.
* Make sure your **webcam is enabled** and accessible.

---

## 🛡️ Disclaimer

This tool is meant for educational and demonstration purposes. It is not a certified exam proctoring solution. False positives may occur.

---

## 🧑‍💻 Author

**YashVardhan Thanvi**
[GitHub](https://github.com/Thanatos9404) • [LinkedIn](https://linkedin.com/in/yashvardhan-thanvi-2a3a661a8)

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

```
