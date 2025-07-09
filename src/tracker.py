import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox
        self.track_id = track_id
        self.hits = 1
        self.no_losses = 0
        self.trace = deque(maxlen=20)

class Sort:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.track_id_count = 0

    def update(self, dets):
        # dets: [[x1, y1, x2, y2, conf], ...]
        updated_tracks = []
        for det in dets:
            matched = False
            for trk in self.trackers:
                if self.iou(det[:4], trk.bbox) > self.iou_threshold:
                    trk.bbox = det[:4]
                    trk.hits += 1
                    trk.no_losses = 0
                    matched = True
                    updated_tracks.append(trk)
                    break
            if not matched:
                self.track_id_count += 1
                new_trk = Track(det[:4], self.track_id_count)
                updated_tracks.append(new_trk)
        # Age tracks
        for trk in self.trackers:
            if trk not in updated_tracks:
                trk.no_losses += 1
                if trk.no_losses < self.max_age:
                    updated_tracks.append(trk)
        self.trackers = [trk for trk in updated_tracks if trk.no_losses < self.max_age]
        return [(trk.bbox, trk.track_id) for trk in self.trackers]

    @staticmethod
    def iou(bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                  (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o 