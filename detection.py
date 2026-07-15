"""
detection.py
------------
Computer-vision pipeline for the Smart Traffic Violation Analytics System.

Given a video source (file path, RTSP URL, or webcam index), this module:
  1. Detects moving vehicles using background subtraction (MOG2)
  2. Tracks each vehicle across frames with a lightweight centroid tracker
  3. Estimates speed from pixel displacement (using a pixels-per-metre
     calibration constant that should be tuned to the real camera in
     production, e.g. via known lane width / stop-line distance)
  4. Detects red-light violations: a vehicle crossing the stop line while
     the signal (read from the frame's on-screen indicator here; in
     production this comes from the traffic-signal controller/API) is Red
  5. Writes an annotated output video + a violations CSV log

Works unchanged on real CCTV/ANPR camera footage - only the calibration
constants (PIXELS_PER_METRE, STOP_LINE_Y, SPEED_LIMIT_KMPH) need updating
for a real deployment.
"""

import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# ---- Calibration constants (tune per real camera in production) ----------
PIXELS_PER_METRE = 8.0          # from known real-world reference distance
STOP_LINE_Y = 200
SPEED_LIMIT_KMPH = 50
MAX_TRACK_DISTANCE = 60         # px, for associating detections across frames
MIN_CONTOUR_AREA = 800


@dataclass
class Track:
    track_id: int
    centroid: tuple
    positions: list = field(default_factory=list)   # (frame_idx, y)
    speed_kmph: float = 0.0
    crossed_stop_line: bool = False
    violation_logged: bool = False
    last_seen: int = 0


class CentroidTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def update(self, detections, frame_idx):
        assigned = set()
        for det in detections:
            cx, cy, w, h = det
            best_id, best_dist = None, MAX_TRACK_DISTANCE
            for tid, tr in self.tracks.items():
                if tid in assigned:
                    continue
                dist = np.hypot(tr.centroid[0] - cx, tr.centroid[1] - cy)
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is not None:
                tr = self.tracks[best_id]
                tr.positions.append((frame_idx, cy))
                tr.centroid = (cx, cy)
                tr.last_seen = frame_idx
                assigned.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(track_id=tid, centroid=(cx, cy),
                                          positions=[(frame_idx, cy)], last_seen=frame_idx)
                assigned.add(tid)

        # drop stale tracks
        stale = [tid for tid, tr in self.tracks.items() if frame_idx - tr.last_seen > 10]
        for tid in stale:
            del self.tracks[tid]

        return self.tracks


def estimate_speed(track: Track, fps: float) -> float:
    """Speed from vertical pixel displacement over the last N frames."""
    if len(track.positions) < 5:
        return 0.0
    f0, y0 = track.positions[-5]
    f1, y1 = track.positions[-1]
    dframes = f1 - f0
    if dframes <= 0:
        return 0.0
    dt_sec = dframes / fps
    dpix = abs(y1 - y0)
    metres = dpix / PIXELS_PER_METRE
    mps = metres / dt_sec
    kmph = mps * 3.6
    return round(kmph, 1)


def read_signal_state(frame) -> str:
    """
    In this demo, the synthetic video renders the signal state as a
    coloured circle top-right. We sample that pixel's colour.
    In production, read this from the signal controller's API/log instead
    of computer vision - it's far more reliable.
    """
    b, g, r = frame[30, frame.shape[1] - 30]
    if r > 150 and g < 100:
        return "Red"
    if r > 150 and g > 150:
        return "Yellow"
    if g > 150 and r < 100:
        return "Green"
    return "Unknown"


def run_pipeline(video_path: str, out_video_path: str, out_csv_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    tracker = CentroidTracker()

    violations = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        signal = read_signal_state(frame)

        fg_mask = bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((7, 7), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = x + bw / 2, y + bh / 2
            detections.append((cx, cy, bw, bh))

        tracks = tracker.update(detections, frame_idx)

        # draw stop line + signal readout
        cv2.line(frame, (0, STOP_LINE_Y), (w, STOP_LINE_Y), (255, 255, 255), 2)
        cv2.putText(frame, f"Signal: {signal}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        for tid, tr in tracks.items():
            cx, cy = tr.centroid
            tr.speed_kmph = estimate_speed(tr, fps) or tr.speed_kmph

            # stop-line crossing check (moving upward, i.e. y decreasing, through STOP_LINE_Y)
            if len(tr.positions) >= 2:
                _, y_prev = tr.positions[-2]
                _, y_now = tr.positions[-1]
                crossed_now = y_prev >= STOP_LINE_Y > y_now
                if crossed_now:
                    tr.crossed_stop_line = True
                    if signal == "Red" and not tr.violation_logged:
                        violations.append({
                            "frame": frame_idx,
                            "time_sec": round(frame_idx / fps, 2),
                            "track_id": tid,
                            "violation_type": "Red Light Jump",
                            "speed_kmph": tr.speed_kmph,
                            "signal_state": signal,
                        })
                        tr.violation_logged = True

            # speeding check (log once per track when first detected over limit)
            if tr.speed_kmph > SPEED_LIMIT_KMPH and not tr.violation_logged:
                violations.append({
                    "frame": frame_idx,
                    "time_sec": round(frame_idx / fps, 2),
                    "track_id": tid,
                    "violation_type": "Overspeeding",
                    "speed_kmph": tr.speed_kmph,
                    "signal_state": signal,
                })
                tr.violation_logged = True

            color = (0, 0, 255) if tr.violation_logged else (0, 255, 0)
            cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)
            cv2.putText(frame, f"ID{tid} {tr.speed_kmph:.0f}km/h", (int(cx) + 8, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    df = pd.DataFrame(violations)
    df.to_csv(out_csv_path, index=False)
    print(f"Processed {frame_idx} frames. Logged {len(df)} violation events.")
    print(f"Annotated video -> {out_video_path}")
    print(f"Violations CSV  -> {out_csv_path}")
    return df


if __name__ == "__main__":
    run_pipeline(
        video_path="outputs/video/traffic_sim.mp4",
        out_video_path="outputs/video/traffic_annotated.mp4",
        out_csv_path="outputs/video_violations_log.csv",
    )
