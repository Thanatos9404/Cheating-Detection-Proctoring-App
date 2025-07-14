import cv2
import mediapipe as mp
import datetime
import numpy as np
import os
import sounddevice as sd
from scipy.io.wavfile import write as write_wav


os.makedirs("cheating_screenshots", exist_ok=True)
os.makedirs("cheating_clips", exist_ok=True)
os.makedirs("cheating_audio", exist_ok=True)

fps = 20
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = None
recording_timer = 0
RECORD_SECONDS = 10
sample_rate = 44100


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

motion_threshold = 800000
cheating_log = []

ret, frame1 = cap.read()
ret, frame2 = cap.read()

font = cv2.FONT_HERSHEY_SIMPLEX
mouth_open_counter = 0
MOUTH_FRAMES_THRESHOLD = 8
lip_distances = []


def capture_cheating_event(frame):
    global video_writer, recording_timer

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    cv2.imwrite(f"cheating_screenshots/screen_{timestamp}.png", frame)

    video_writer = cv2.VideoWriter(f"cheating_clips/video_{timestamp}.avi", fourcc, fps, (frame.shape[1], frame.shape[0]))
    recording_timer = fps * RECORD_SECONDS

    def record_audio():
        audio = sd.rec(int(RECORD_SECONDS * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        write_wav(f"cheating_audio/audio_{timestamp}.wav", sample_rate, audio)

    import threading
    threading.Thread(target=record_audio).start()



def log_event(reason):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log = f"{timestamp} - {reason}"
    if not cheating_log or cheating_log[-1] != log:
        cheating_log.append(log)
        print(log)

def get_head_pose(landmarks, frame):
    image_points = np.array([
        landmarks[0],
        landmarks[1],
        landmarks[2],
        landmarks[3],
        landmarks[4],
        landmarks[5]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (-30.0, -30.0, -30.0),
        (30.0, -30.0, -30.0),
        (-40.0, 30.0, -30.0),
        (40.0, 30.0, -30.0),
        (0.0, 70.0, -50.0)
    ])

    size = frame.shape
    focal_length = size[1]
    center = (size[1] // 2, size[0] // 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cheating_flag = False

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_detection.process(rgb)
    face_count = 0

    if face_result.detections:
        for d in face_result.detections:
            if d.location_data.relative_bounding_box.width > 0:
                if d.score[0] < 0.5:
                    log_event("Face visibility low / covered")
                    cv2.putText(frame, "Face Hidden!", (20, 50), font, 0.8, (0, 0, 255), 2)
                    cheating_flag = True
                    if recording_timer <= 0:
                        capture_cheating_event(frame)

        face_count = len(face_result.detections)

    if face_count > 1:
        log_event("Multiple faces detected")
        cv2.putText(frame, "Multiple People Detected!", (20, 80), font, 0.8, (0, 0, 255), 2)
        cheating_flag = True
        if recording_timer <= 0:
            capture_cheating_event(frame)

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=2)
    motion = cv2.countNonZero(dilated)

    if motion > motion_threshold:
        log_event("Significant motion detected")
        cv2.putText(frame, "❌ Suspicious Movement!", (20, 110), font, 0.8, (0, 0, 255), 2)
        cheating_flag = True
        if recording_timer <= 0:
            capture_cheating_event(frame)

    mesh_results = face_mesh.process(rgb)
    if mesh_results.multi_face_landmarks:
        for landmarks in mesh_results.multi_face_landmarks:
            lm = landmarks.landmark
            visible_landmarks = len(lm)

            if visible_landmarks < 460:
                log_event("Face might be covered (landmarks missing)")
                cv2.putText(frame, "❌ Face Obstructed!", (20, 50), font, 0.8, (0, 0, 255), 2)
                cheating_flag = True
                if recording_timer <= 0:
                    capture_cheating_event(frame)

            try:
                ids = [1, 33, 263, 61, 291, 199]
                coords = [(int(lm[i].x * w), int(lm[i].y * h)) for i in ids if i < len(lm)]
                if len(coords) == 6:
                    rot_vec = get_head_pose(coords, frame)
                    rot_deg = rot_vec[1][0] * 57.3
                    if rot_deg < -25:
                        log_event("Looking too much left")
                        cv2.putText(frame, "❌ Looking Away (Left)", (20, 80), font, 0.8, (0, 0, 255), 2)
                        cheating_flag = True
                        if recording_timer <= 0:
                            capture_cheating_event(frame)

                    elif rot_deg > 25:
                        log_event("Looking too much right")
                        cv2.putText(frame, "❌ Looking Away (Right)", (20, 80), font, 0.8, (0, 0, 255), 2)
                        cheating_flag = True
                        if recording_timer <= 0:
                            capture_cheating_event(frame)

            except:
                pass

            try:
                upper_lip = lm[13]
                lower_lip = lm[14]
                dist = abs((lower_lip.y - upper_lip.y) * h)
                lip_distances.append(dist)
                if len(lip_distances) > 6:
                    lip_distances.pop(0)
                avg_dist = sum(lip_distances) / len(lip_distances)

                if avg_dist > 1.5:
                    mouth_open_counter += 1
                else:
                    mouth_open_counter = 0

                if mouth_open_counter > MOUTH_FRAMES_THRESHOLD:
                    log_event("Talking detected")
                    cv2.putText(frame, "Talking Detected", (20, 110), font, 0.8, (0, 0, 255), 2)
                    cheating_flag = True
                    if recording_timer <= 0:
                        capture_cheating_event(frame)

            except:
                pass
    else:
        log_event("Face not detected")
        cv2.putText(frame, "❌ Face Missing!", (20, 50), font, 0.8, (0, 0, 255), 2)
        cheating_flag = True
        if recording_timer <= 0:
            capture_cheating_event(frame)

    frame1 = frame2
    ret, frame2 = cap.read()

    if not cheating_flag:
        cv2.putText(frame, "Monitoring... ✅", (20, 30), font, 0.8, (0, 255, 0), 2)

    if video_writer and recording_timer > 0:
        video_writer.write(frame)
        recording_timer -= 1
        if recording_timer <= 0:
            video_writer.release()
            video_writer = None

    cv2.imshow("Online Exam Proctoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open("cheating_log.txt", "w") as f:
    for entry in cheating_log:
        f.write(entry + "\n")

cap.release()
cv2.destroyAllWindows()
