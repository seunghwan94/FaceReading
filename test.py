import face_recognition
import math
from statistics import mean
import numpy as np
import json
import cv2

# ── Utility Functions ─────────────────────────────────────────────────────────
def centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (mean(xs), mean(ys))

def euclidean_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def poly_length(pts):
    return sum(euclidean_dist(pts[i], pts[i+1]) for i in range(len(pts)-1))

def triangle_area(a, b, c):
    return abs((a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))/2)

# ── 1) Load Image & Detect Landmarks ─────────────────────────────────────────
image_path = "./img/test.jpg"
image = face_recognition.load_image_file(image_path)
face_locs = face_recognition.face_locations(image)
lm_list = face_recognition.face_landmarks(image)
if not face_locs or not lm_list:
    raise RuntimeError("얼굴을 찾을 수 없습니다.")

top, right, bottom, left = face_locs[0]
fw, fh = right - left, bottom - top
lm = lm_list[0]

# ── 2) Extract Points by Region ──────────────────────────────────────────────
left_eye_pts   = lm["left_eye"]
right_eye_pts  = lm["right_eye"]
left_eb_pts    = lm["left_eyebrow"]
right_eb_pts   = lm["right_eyebrow"]
nose_bridge    = lm["nose_bridge"]
nose_tip_pts   = lm["nose_tip"]
top_lip_pts    = lm["top_lip"]
bottom_lip_pts = lm["bottom_lip"]
chin_pts       = lm["chin"]

# ── 3) Compute Core Keypoints ─────────────────────────────────────────────────
left_eye_cent  = centroid(left_eye_pts)
right_eye_cent = centroid(right_eye_pts)
nose_tip_cent  = centroid(nose_tip_pts)
mouth_cent     = centroid(top_lip_pts + bottom_lip_pts)
chin_bottom    = chin_pts[len(chin_pts)//2]

# ── 4) Primary Distance Metrics ──────────────────────────────────────────────
metrics = {
    "face_width": fw,
    "face_height": fh,
    "eye_distance":     euclidean_dist(left_eye_cent, right_eye_cent),
    "left_eye_width":   euclidean_dist(left_eye_pts[0], left_eye_pts[-1]),
    "right_eye_width":  euclidean_dist(right_eye_pts[0], right_eye_pts[-1]),
    "left_eye_height":  euclidean_dist(left_eye_pts[1], left_eye_pts[5]),
    "right_eye_height": euclidean_dist(right_eye_pts[1], right_eye_pts[5]),
    "nose_length":      euclidean_dist(nose_bridge[0], nose_tip_pts[-1]),
    "nose_width":       euclidean_dist(nose_tip_pts[0], nose_tip_pts[-1]),
    "mouth_width":      euclidean_dist(top_lip_pts[0], top_lip_pts[6]),
    "mouth_height":     euclidean_dist(centroid(top_lip_pts), centroid(bottom_lip_pts)),
    "eye_to_mouth":     euclidean_dist(
                             ((left_eye_cent[0]+right_eye_cent[0])/2,
                              (left_eye_cent[1]+right_eye_cent[1])/2),
                             mouth_cent),
    "eye_left_to_chin":  euclidean_dist(left_eye_cent, chin_bottom),
    "eye_right_to_chin": euclidean_dist(right_eye_cent, chin_bottom),
    "nose_to_mouth":    euclidean_dist(nose_tip_cent, mouth_cent),
    "nose_to_chin":     euclidean_dist(nose_tip_cent, chin_bottom),
    "jaw_width":        euclidean_dist(chin_pts[0], chin_pts[-1]),
    "jaw_length":       poly_length(chin_pts)
}

# ── 5) 20 Additional Metrics ─────────────────────────────────────────────────
# 1. Mouth tilt
metrics["mouth_tilt_angle"] = math.degrees(math.atan2(
    top_lip_pts[6][1] - top_lip_pts[0][1],
    top_lip_pts[6][0] - top_lip_pts[0][0]
))
# 2-3. Eyebrow-to-eye vertical distance
metrics["left_eb_to_eye_dist"]  = abs(centroid(left_eb_pts)[1] - left_eye_cent[1])
metrics["right_eb_to_eye_dist"] = abs(centroid(right_eb_pts)[1] - right_eye_cent[1])
# 4. Eye-mouth axis difference
eye_axis   = math.degrees(math.atan2(
    right_eye_cent[1] - left_eye_cent[1],
    right_eye_cent[0] - left_eye_cent[0]
))
mouth_axis = math.degrees(math.atan2(
    top_lip_pts[6][1] - top_lip_pts[0][1],
    top_lip_pts[6][0] - top_lip_pts[0][0]
))
metrics["eye_mouth_axis_diff"] = abs(eye_axis - mouth_axis)
# 5. Nose-chin vector angle vs vertical
nc_vec = (nose_tip_cent[0] - chin_bottom[0], nose_tip_cent[1] - chin_bottom[1])
metrics["nose_chin_vector_angle"] = math.degrees(math.acos(
    nc_vec[1] / math.hypot(nc_vec[0], nc_vec[1])
))
# 6. Cheek-to-jaw ratios & asymmetry
cheek1 = euclidean_dist(left_eye_cent, chin_pts[0])
cheek2 = euclidean_dist(right_eye_cent, chin_pts[-1])
metrics["cheek_to_jaw_ratio"]   = mean([cheek1, cheek2]) / metrics["jaw_width"]
metrics["cheek_asymmetry"]      = abs(cheek1 - cheek2)
# 7-8. Eyebrow-length to cheek ratios
metrics["left_eb_cheek_ratio"]  = poly_length(left_eb_pts) / cheek1
metrics["right_eb_cheek_ratio"] = poly_length(right_eb_pts) / cheek2
# 9. Nose bridge length
metrics["nose_bridge_length"] = poly_length(nose_bridge)
# 10. Philtrum triangle area
metrics["philtrum_triangle_area"] = triangle_area(
    nose_tip_cent,
    tuple(top_lip_pts[0]),
    tuple(top_lip_pts[6])
)
# 11. Face diagonal
metrics["face_diagonal"] = math.hypot(fw, fh)
# 12-13. Convex hull area & perimeter
all_pts = np.array(left_eye_pts + right_eye_pts + left_eb_pts + right_eb_pts +
                   nose_bridge + nose_tip_pts + top_lip_pts + bottom_lip_pts + chin_pts)
hull = cv2.convexHull(all_pts)
metrics["hull_area"]      = cv2.contourArea(hull)
metrics["hull_perimeter"] = cv2.arcLength(hull, True)
# 14. Bounding box area
metrics["bounding_box_area"] = fw * fh
# 15-16. Eyebrow contour areas
metrics["left_eb_area"]  = cv2.contourArea(np.array(left_eb_pts))
metrics["right_eb_area"] = cv2.contourArea(np.array(right_eb_pts))
# 17-18. Eyebrow height diff
metrics["left_eb_height_diff"]  = abs(left_eb_pts[0][1] - left_eb_pts[-1][1])
metrics["right_eb_height_diff"] = abs(right_eb_pts[0][1] - right_eb_pts[-1][1])
# 19-20. Eye-to-nose distances
metrics["left_eye_to_nose_dist"]  = euclidean_dist(left_eye_cent, nose_tip_cent)
metrics["right_eye_to_nose_dist"] = euclidean_dist(right_eye_cent, nose_tip_cent)

# ── 6) Output JSON ────────────────────────────────────────────────────────────
print(json.dumps(metrics, indent=2, ensure_ascii=False))
with open("metrics_full.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# ── 7) Visualization ─────────────────────────────────────────────────────────
img = cv2.imread(image_path)
mapping = {
    "eye_distance":           (left_eye_cent, right_eye_cent, (0,255,0)),
    "left_eye_width":         (left_eye_pts[0], left_eye_pts[-1], (0,200,0)),
    "right_eye_width":        (right_eye_pts[0], right_eye_pts[-1], (0,200,0)),
    "left_eye_height":        (left_eye_pts[1], left_eye_pts[5], (0,150,0)),
    "right_eye_height":       (right_eye_pts[1], right_eye_pts[5], (0,150,0)),
    "nose_length":            (nose_bridge[0], nose_tip_pts[-1], (0,0,255)),
    "nose_width":             (nose_tip_pts[0], nose_tip_pts[-1], (0,0,200)),
    "mouth_width":            (top_lip_pts[0], top_lip_pts[6], (255,0,0)),
    "mouth_height":           (centroid(top_lip_pts), centroid(bottom_lip_pts), (200,0,0)),
    "eye_to_mouth":           (((left_eye_cent[0]+right_eye_cent[0])/2,
                                (left_eye_cent[1]+right_eye_cent[1])/2),
                               mouth_cent, (0,255,255)),
    "eye_left_to_chin":       (left_eye_cent, chin_bottom, (0,200,200)),
    "eye_right_to_chin":      (right_eye_cent, chin_bottom, (0,200,200)),
    "nose_to_mouth":          (nose_tip_cent, mouth_cent, (255,255,0)),
    "nose_to_chin":           (nose_tip_cent, chin_bottom, (255,200,0)),
    "jaw_width":              (chin_pts[0], chin_pts[-1], (150,150,150)),
    "jaw_length":             (chin_pts, None, (100,100,100)),
    "mouth_tilt_angle":       (top_lip_pts[0], top_lip_pts[6], (0,255,255)),
    "left_eb_to_eye_dist":    (centroid(left_eb_pts), left_eye_cent, (255,0,255)),
    "right_eb_to_eye_dist":   (centroid(right_eb_pts), right_eye_cent, (255,0,255)),
    "eye_mouth_axis_diff":    (left_eye_cent, right_eye_cent, (0,200,255)),
    "mouth_axis":             (top_lip_pts[0], top_lip_pts[6], (200,200,0)),
    "nose_chin_vector_angle": (nose_tip_cent, chin_bottom, (0,0,200)),
    "cheek_to_jaw_ratio":     (left_eye_cent, chin_pts[0], (255,255,0)),
    "cheek_asymmetry":        (right_eye_cent, chin_pts[-1], (255,255,0)),
    "left_eb_cheek_ratio":    (left_eb_pts, None, (0,255,150)),
    "right_eb_cheek_ratio":   (right_eb_pts, None, (0,255,150)),
    "nose_bridge_length":     (nose_bridge, None, (255,0,255)),
    "philtrum_triangle_area": (nose_tip_cent, None, (0,255,0)),
    "face_diagonal":          ((left, top), (right, bottom), (100,100,100)),
    "hull":                   (hull.squeeze(), None, (0,0,0)),
    "bounding_box_area":      ((left, top), (right, bottom), (0,0,0)),
    "left_eb_area":           (left_eb_pts, None, (255,150,0)),
    "right_eb_area":          (right_eb_pts, None, (255,150,0)),
    "left_eb_height_diff":    (left_eb_pts[0], left_eb_pts[-1], (0,150,255)),
    "right_eb_height_diff":   (right_eb_pts[0], right_eb_pts[-1], (0,150,255)),
    "left_eye_to_nose_dist":  (left_eye_cent, nose_tip_cent, (150,0,150)),
    "right_eye_to_nose_dist": (right_eye_cent, nose_tip_cent, (150,0,150))
}
# Draw
for key, (p1, p2, col) in mapping.items():
    if isinstance(p1, list) or isinstance(p1, np.ndarray):
        pts = np.array([tuple(map(int, pt)) for pt in p1])
        cv2.polylines(img, [pts], True, col, 2)
        mid = tuple(pts[len(pts)//2])
    else:
        pt1 = tuple(map(int, p1))
        pt2 = tuple(map(int, p2)) if p2 is not None else pt1
        cv2.line(img, pt1, pt2, col, 2)
        mid = ((pt1[0]+pt2[0])//2, (pt1[1]+pt2[1])//2)
    val = metrics.get(key)
    if val is not None:
        cv2.putText(img, f"{key}:{val:.1f}", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
# Save
cv2.imwrite("annotated_all_metrics_full.jpg", img)
print("Annotated image saved to 'annotated_all_metrics_full.jpg'")
