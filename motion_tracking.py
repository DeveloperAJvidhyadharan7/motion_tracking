import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

def choose_file():
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Drone Video",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    )

def track_motion_with_trails(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Cannot open video.")
        return

    # Feature detectors
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Error: Cannot read first frame.")
        return

    height, width = prev_frame.shape[:2]
    trail_mask = np.zeros((height, width, 3), dtype=np.uint8)  # For drawing trails

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kp = fast.detect(prev_gray, None)
    prev_kp, prev_des = orb.compute(prev_gray, prev_kp)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = fast.detect(gray, None)
        kp, des = orb.compute(gray, kp)

        if prev_des is not None and des is not None:
            matches = bf.match(prev_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            for m in matches[:50]:  # Limit to top 50 matches
                pt1 = tuple(map(int, prev_kp[m.queryIdx].pt))
                pt2 = tuple(map(int, kp[m.trainIdx].pt))

                # Draw fading trails
                cv2.line(trail_mask, pt1, pt2, (0, 255, 255), 2)
                cv2.circle(trail_mask, pt2, 2, (0, 128, 255), -1)

        # Fade trail effect
        trail_mask = (trail_mask * 0.9).astype(np.uint8)

        # Combine trails with current frame
        output = cv2.addWeighted(frame, 0.8, trail_mask, 0.6, 0)

        cv2.imshow("🔥 Enhanced Motion Tracking (Trails Effect)", output)

        if cv2.waitKey(30) & 0xFF == 27:  # ESC key
            break

        prev_gray = gray
        prev_kp = kp
        prev_des = des

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = choose_file()
    if video_path and os.path.exists(video_path):
        print(f"📁 Selected video: {video_path}")
        track_motion_with_trails(video_path)
    else:
        print("❌ No valid file selected.")
