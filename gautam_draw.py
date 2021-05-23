import cv2
font = cv2.FONT_HERSHEY_PLAIN
org = (50, 50)
fontScale = 1
thickness=2
color = (255, 0, 0)
def draw_bounding_boxes(tracker,detections,img):

    for track, det in zip(tracker.tracks, detections):
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        trk_bbox = track.to_tlbr()
        det_bbox = det.to_tlbr()
        cv2.rectangle(img, (int(trk_bbox[0]), int(trk_bbox[1])), (int(trk_bbox[2]), int(trk_bbox[3])),(255, 255, 255), 2)
        cv2.rectangle(img, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])), (255, 0, 0), 2)
        cv2.putText(img, 'helmet_id: '+str(track.track_id), (int(trk_bbox[0]), int(trk_bbox[1])),0, 5e-3 * 200, (0, 0, 0), 2)