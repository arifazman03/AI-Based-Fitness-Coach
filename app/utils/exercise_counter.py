import cv2
import mediapipe as mp
import numpy as np

class ExerciseCounter:
    def __init__(self, exercise):
        self.exercise = exercise
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_draw = mp.solutions.drawing_utils
        self.count = 0
        self.stage = None

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180 else angle

    def get_keypoints(self, landmarks):
        lm = self.mp_pose.PoseLandmark
        mapping = {
            "bicep_curl": [lm.LEFT_SHOULDER, lm.LEFT_ELBOW, lm.LEFT_WRIST],
            "shoulder_press": [lm.LEFT_HIP, lm.LEFT_SHOULDER, lm.LEFT_ELBOW],
            "squat": [lm.LEFT_HIP, lm.LEFT_KNEE, lm.LEFT_ANKLE]
        }
        return mapping.get(self.exercise)

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            keypoints = self.get_keypoints(landmarks)

            if keypoints:
                a = [landmarks[keypoints[0].value].x, landmarks[keypoints[0].value].y]
                b = [landmarks[keypoints[1].value].x, landmarks[keypoints[1].value].y]
                c = [landmarks[keypoints[2].value].x, landmarks[keypoints[2].value].y]

                angle = self.calculate_angle(a, b, c)

                # Rep logic
                if angle > 160:
                    self.stage = "down"
                if angle < 30 and self.stage == "down":
                    self.stage = "up"
                    self.count += 1

        cv2.putText(image, f'Reps: {self.count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image, self.count
