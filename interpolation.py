import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from filterpy.kalman import KalmanFilter

###############################################################################
#                            UPDATED BYTETRACK CLASS                          #
###############################################################################

from typing import List, Tuple
from supervision.detection.core import Detections
from supervision.detection.utils import box_iou_batch
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.kalman_filter import KalmanFilter as ByteTrackKalmanFilter
from supervision.tracker.byte_tracker.single_object_track import STrack, TrackState
from supervision.tracker.byte_tracker.utils import IdCounter


def joint_tracks(track_list_a: List[STrack], track_list_b: List[STrack]) -> List[STrack]:
    seen_track_ids = set()
    result = []
    for track in track_list_a + track_list_b:
        if track.internal_track_id not in seen_track_ids:
            seen_track_ids.add(track.internal_track_id)
            result.append(track)
    return result

def sub_tracks(track_list_a: List[STrack], track_list_b: List[STrack]) -> List[int]:
    tracks = {track.internal_track_id: track for track in track_list_a}
    track_ids_b = {track.internal_track_id for track in track_list_b}
    for track_id in track_ids_b:
        tracks.pop(track_id, None)
    return list(tracks.values())

def remove_duplicate_tracks(
    tracks_a: List[STrack], tracks_b: List[STrack]
) -> Tuple[List[STrack], List[STrack]]:
    pairwise_distance = matching.iou_distance(tracks_a, tracks_b)
    matching_pairs = np.where(pairwise_distance < 0.15)

    duplicates_a, duplicates_b = set(), set()
    for track_index_a, track_index_b in zip(*matching_pairs):
        time_a = tracks_a[track_index_a].frame_id - tracks_a[track_index_a].start_frame
        time_b = tracks_b[track_index_b].frame_id - tracks_b[track_index_b].start_frame
        if time_a > time_b:
            duplicates_b.add(track_index_b)
        else:
            duplicates_a.add(track_index_a)

    result_a = [track for i, track in enumerate(tracks_a) if i not in duplicates_a]
    result_b = [track for i, track in enumerate(tracks_b) if i not in duplicates_b]
    return result_a, result_b


class ByteTrack:
    """
    Updated ByteTrack class with 'track_buffer' parameter.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        track_buffer: int = 30,  # Replaces old lost_track_buffer
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
    ):
        self.track_activation_threshold = track_activation_threshold
        self.minimum_matching_threshold = minimum_matching_threshold

        # For internal logic
        self.frame_id = 0
        self.det_thresh = self.track_activation_threshold + 0.1
        self.max_time_lost = int(frame_rate / 30.0 * track_buffer)  # scaling lost frames
        self.minimum_consecutive_frames = minimum_consecutive_frames

        # Kalman filters used inside ByteTrack
        self.kalman_filter = ByteTrackKalmanFilter()
        self.shared_kalman = ByteTrackKalmanFilter()

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []

        self.internal_id_counter = IdCounter()
        self.external_id_counter = IdCounter(start_id=1)

    def reset(self) -> None:
        self.frame_id = 0
        self.internal_id_counter.reset()
        self.external_id_counter.reset()
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []

    def update_with_detections(self, detections: Detections) -> Detections:
        """
        Updates the tracker with the given detections and returns
        only the ones that have an assigned tracker ID.
        """
        # If no detections, just update internals and return empty
        if len(detections) == 0:
            self.update_with_tensors(np.zeros((0, 5), dtype=float))
            empty_dets = Detections.empty()
            empty_dets.tracker_id = np.array([], dtype=int)
            return empty_dets

        # Nx5 array -> [x1, y1, x2, y2, confidence]
        tensors = np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))

        # Update ByteTrack, returning active STrack objects
        tracks = self.update_with_tensors(tensors)

        if len(tracks) == 0:
            empty_dets = Detections.empty()
            empty_dets.tracker_id = np.array([], dtype=int)
            return empty_dets

        # Assign track IDs back to detections
        detection_bboxes = np.asarray([det[:4] for det in tensors])
        track_bboxes = np.asarray([track.tlbr for track in tracks])

        ious = box_iou_batch(detection_bboxes, track_bboxes)
        iou_costs = 1 - ious
        matches, _, _ = matching.linear_assignment(iou_costs, 0.5)

        detections.tracker_id = np.full(len(detections), -1, dtype=int)
        for i_det, i_trk in matches:
            detections.tracker_id[i_det] = int(tracks[i_trk].external_track_id)

        return detections[detections.tracker_id != -1]

    def update_with_tensors(self, tensors: np.ndarray) -> List[STrack]:
        """
        Core ByteTrack update logic, with Nx5 [x1,y1,x2,y2,confidence].
        Returns the list of 'active' STrack objects.
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = tensors[:, 4] if len(tensors) > 0 else np.array([])
        bboxes = tensors[:, :4] if len(tensors) > 0 else np.zeros((0, 4))

        remain_inds = scores > self.track_activation_threshold
        inds_low = scores > 0.1
        inds_high = scores < self.track_activation_threshold
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        # Build STrack objects for main detections
        if len(dets) > 0:
            detections = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_keep,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                )
                for (tlbr, score_keep) in zip(dets, scores_keep)
            ]
        else:
            detections = []

        # Separate unconfirmed vs tracked
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 1: match with high-confidence boxes
        strack_pool = joint_tracks(tracked_stracks, self.lost_tracks)
        STrack.multi_predict(strack_pool, self.shared_kalman)  # predict new positions
        dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.minimum_matching_threshold
        )
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        # Step 2: match with lower-confidence boxes
        if len(dets_second) > 0:
            detections_second = [
                STrack(
                    STrack.tlbr_to_tlwh(tlbr),
                    score_second,
                    self.minimum_consecutive_frames,
                    self.shared_kalman,
                    self.internal_id_counter,
                    self.external_id_counter,
                )
                for (tlbr, score_second) in zip(dets_second, scores_second)
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.state = TrackState.Lost
                lost_stracks.append(track)

        # Step 3: handle unconfirmed
        detections_unmatched = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections_unmatched)
        dists = matching.fuse_score(dists, detections_unmatched)
        matches, u_unconfirmed, u_detection_new = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_unmatched[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = TrackState.Removed
            removed_stracks.append(track)

        # Step 4: add new stracks
        for inew in u_detection_new:
            track = detections_unmatched[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # Step 5: update lost / removed
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.state = TrackState.Removed
                removed_stracks.append(track)

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks = joint_tracks(self.tracked_tracks, activated_starcks)
        self.tracked_tracks = joint_tracks(self.tracked_tracks, refind_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost_stracks)
        self.lost_tracks = sub_tracks(self.lost_tracks, self.removed_tracks)
        self.removed_tracks = removed_stracks
        self.tracked_tracks, self.lost_tracks = remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )

        output_stracks = [track for track in self.tracked_tracks if track.is_activated]
        return output_stracks


###############################################################################
#                        MAIN CODE USING YOLO + BYTETRACK                     #
###############################################################################

# ---------------------- Configuration ----------------------
YOLO_MODEL_PATH = r"C:\Users\admin\Downloads\car_detection\HELIPAD_DETECTION\best.pt"
CONFIDENCE_THRESHOLD = 0.4
VIDEO_SOURCE = r"C:\Users\admin\Downloads\car_detection\HELIPAD_DETECTION\video.mp4"
TRACK_BUFFER_SIZE = 30  # Keep lost tracks around longer to see predictions

# ---------------------- Initialize ByteTrack ----------------
# (this is OUR updated ByteTrack class above, with track_buffer)
tracker = ByteTrack(
    track_activation_threshold=0.25,
    track_buffer=TRACK_BUFFER_SIZE,  # <-- the key parameter
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=1
)

# ---------------------- Initialize YOLO & Annotators -------
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

model = YOLO(YOLO_MODEL_PATH)
print("Loaded YOLO model with classes:", model.names)

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Error: Unable to open video source {VIDEO_SOURCE}")
    exit()

# ---------------------- Kalman Filter For Interpolation ----
kalman_filters = {}  # track_id -> KalmanFilter
last_bboxes = {}     # track_id -> last known bounding box

def create_kalman_filter():
    """
    Basic 2D Kalman for (x, y, vx, vy).
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=float)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=float)
    kf.P *= 10.0
    kf.R *= 1.0
    kf.Q *= 0.01
    return kf

def init_kf_for_track(track_id, bbox):
    """
    Initialize the filter for a new track.
    bbox = [x1, y1, x2, y2]
    """
    kf = create_kalman_filter()
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0
    kf.x = np.array([[x_center], [y_center], [0], [0]], dtype=float)
    return kf

def update_kf(kf, bbox=None):
    """
    kf.predict()
    if bbox is not None: kf.update(measurement)
    """
    kf.predict()
    if bbox is not None:
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        measurement = np.array([x_center, y_center], dtype=float)
        kf.update(measurement)
    return kf.x[0, 0], kf.x[1, 0]

def predict_bbox(kf, last_bbox):
    """
    Shift the last bbox around the predicted center.
    """
    x1, y1, x2, y2 = last_bbox
    w = x2 - x1
    h = y2 - y1
    x_center_pred, y_center_pred = kf.x[0, 0], kf.x[1, 0]
    x1_pred = x_center_pred - w / 2
    x2_pred = x_center_pred + w / 2
    y1_pred = y_center_pred - h / 2
    y2_pred = y_center_pred + h / 2
    return [float(x1_pred), float(y1_pred), float(x2_pred), float(y2_pred)]

# ---------------------- Main Loop -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    # YOLO expects RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(image_rgb, conf=CONFIDENCE_THRESHOLD)
    yolo_detections = results[0]

    # Convert YOLO -> supervision Detections
    detections = sv.Detections.from_ultralytics(yolo_detections)

    # Update ByteTrack
    updated_detections = tracker.update_with_detections(detections)

    current_frame_track_ids = set()
    all_xyxy = []
    all_conf = []
    all_cls = []
    all_tid = []
    all_pred_flags = []

    # -------- 1) Real Detections
    for i in range(len(updated_detections)):
        track_id = updated_detections.tracker_id[i]
        bbox = updated_detections.xyxy[i]
        cls_id = updated_detections.class_id[i]
        conf = updated_detections.confidence[i]

        current_frame_track_ids.add(track_id)

        # Init KF if new track
        if track_id not in kalman_filters:
            kalman_filters[track_id] = init_kf_for_track(track_id, bbox)
            last_bboxes[track_id] = bbox

        # Update KF
        kf = kalman_filters[track_id]
        update_kf(kf, bbox=bbox)
        last_bboxes[track_id] = bbox

        # Add to annotation arrays
        all_xyxy.append(bbox)
        all_conf.append(conf)
        all_cls.append(cls_id)
        all_tid.append(track_id)
        all_pred_flags.append(False)

    # -------- 2) Predict missing tracks
    for track in tracker.tracked_tracks + tracker.lost_tracks:
        tid = getattr(track, 'track_id', None)
        if tid is None:
            continue

        if tid not in current_frame_track_ids:
            # We have a lost track => do pure KF predict
            if tid in kalman_filters and tid in last_bboxes:
                kf = kalman_filters[tid]
                update_kf(kf, bbox=None)

                predicted_bbox = predict_bbox(kf, last_bboxes[tid])
                pred_conf = getattr(track.last_detection, "confidence", 0.5)
                pred_cls = getattr(track, "class_id", 0)

                all_xyxy.append(predicted_bbox)
                all_conf.append(pred_conf)
                all_cls.append(pred_cls)
                all_tid.append(tid)
                all_pred_flags.append(True)

    # -------- 3) Combine real + predicted for display
    if len(all_xyxy) > 0:
        all_xyxy = np.array(all_xyxy, dtype=float)
        all_conf = np.array(all_conf, dtype=float)
        all_cls = np.array(all_cls, dtype=int)
        all_tid = np.array(all_tid, dtype=int)

        sv_detections = sv.Detections(
            xyxy=all_xyxy,
            confidence=all_conf,
            class_id=all_cls,
            tracker_id=all_tid
        )
    else:
        sv_detections = sv.Detections.empty()
        all_pred_flags = []

    # -------- 4) Annotate
    labels = []
    colors = []
    for i in range(len(sv_detections)):
        cls_id = sv_detections.class_id[i]
        tid = sv_detections.tracker_id[i]
        pred_flag = all_pred_flags[i]

        class_name = model.names.get(int(cls_id), "Unknown")
        base_text = f"ID: {tid} | {class_name}"
        if pred_flag:
            label = f"{base_text} (Predicted)"
            color = (0, 0, 255)  # red for predicted
        else:
            label = base_text
            color = (0, 255, 0)  # green for real
        labels.append(label)
        colors.append(color)

    annotated_frame = box_annotator.annotate(
        scene=image_rgb.copy(),
        detections=sv_detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=sv_detections,
        labels=labels
    )

    # Convert to BGR for display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Optional console print
    for lbl in labels:
        print(lbl)

    cv2.imshow("YOLO + ByteTrack + Kalman Interpolation", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
