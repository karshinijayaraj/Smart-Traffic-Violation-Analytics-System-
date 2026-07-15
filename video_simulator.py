"""
video_simulator.py
-------------------
Generates a short synthetic traffic-camera video so the computer-vision
pipeline (detection.py) can be demonstrated end-to-end without requiring
a licensed real-world CCTV dataset.

Replace this with a real RTSP/CCTV feed or recorded footage in production -
detection.py works on any cv2.VideoCapture source unchanged.

Simulates:
  - A 4-lane road with a stop line
  - Vehicles (rectangles) moving at different speeds
  - A traffic signal that cycles Green -> Yellow -> Red
  - Some vehicles deliberately crossing the stop line during Red (violators)
"""

import cv2
import numpy as np

WIDTH, HEIGHT = 960, 540
FPS = 20
DURATION_SEC = 12
STOP_LINE_Y = 200
LANE_X = [100, 300, 500, 700]
OUT_PATH = "outputs/video/traffic_sim.mp4"

SIGNAL_CYCLE = [("Green", 4), ("Yellow", 1), ("Red", 4), ("Green", 3)]  # seconds


class Vehicle:
    def __init__(self, lane_x, speed_px_per_frame, color, vtype, will_violate=False):
        self.x = lane_x
        self.y = HEIGHT + np.random.randint(0, 150)
        self.speed = speed_px_per_frame
        self.color = color
        self.vtype = vtype
        self.will_violate = will_violate
        self.w, self.h = (36, 60) if vtype != "bus" else (50, 90)

    def step(self):
        self.y -= self.speed

    def rect(self):
        return int(self.x - self.w / 2), int(self.y - self.h / 2), self.w, self.h


def signal_state_at(frame_idx, fps):
    t = (frame_idx / fps) % sum(s for _, s in SIGNAL_CYCLE)
    acc = 0
    for state, dur in SIGNAL_CYCLE:
        acc += dur
        if t < acc:
            return state
    return SIGNAL_CYCLE[-1][0]


def main():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    n_frames = FPS * DURATION_SEC
    vehicles = []
    spawn_every = 12
    rng = np.random.default_rng(7)

    for frame_idx in range(n_frames):
        frame = np.full((HEIGHT, WIDTH, 3), (40, 40, 40), dtype=np.uint8)

        # lane markings
        for lx in [200, 400, 600]:
            for y in range(0, HEIGHT, 30):
                cv2.line(frame, (lx, y), (lx, y + 15), (200, 200, 200), 2)

        signal = signal_state_at(frame_idx, FPS)
        signal_color = {"Green": (0, 200, 0), "Yellow": (0, 200, 200), "Red": (0, 0, 220)}[signal]

        # stop line
        cv2.line(frame, (60, STOP_LINE_Y), (WIDTH - 60, STOP_LINE_Y), (255, 255, 255), 3)
        # signal light
        cv2.circle(frame, (WIDTH - 30, 30), 15, signal_color, -1)
        cv2.putText(frame, f"SIGNAL: {signal}", (WIDTH - 220, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, signal_color, 2)

        # spawn vehicles
        if frame_idx % spawn_every == 0:
            lane_x = int(rng.choice(LANE_X))
            speed = rng.uniform(3, 9)
            vtype = rng.choice(["car", "car", "bus", "car"])
            color = tuple(int(c) for c in rng.integers(80, 255, size=3))
            # occasionally force a fast violator that runs the red light
            will_violate = signal == "Red" and rng.random() < 0.35 and speed > 6
            vehicles.append(Vehicle(lane_x, speed, color, vtype, will_violate))

        for v in vehicles:
            v.step()
            x, y, w, h = v.rect()
            cv2.rectangle(frame, (x, y), (x + w, y + h), v.color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

        vehicles = [v for v in vehicles if v.y > -50]

        writer.write(frame)

    writer.release()
    print(f"Synthetic traffic video written to {OUT_PATH} ({n_frames} frames @ {FPS}fps)")


if __name__ == "__main__":
    main()
