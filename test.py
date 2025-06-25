import face_recognition
import math
from statistics import mean

def centroid(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (mean(xs), mean(ys))

def euclidean_dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def polygon_area(points):
    area = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

def slope_angle(p1, p2):
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))

def vector_angle(a, b):
    dot = a[0]*b[0] + a[1]*b[1]
    mag_a = math.hypot(a[0], a[1])
    mag_b = math.hypot(b[0], b[1])
    return math.degrees(math.acos(dot / (mag_a * mag_b)))

# 1) Load image & detect face + landmarks
image_path = "./img/test.jpg"
image = face_recognition.load_image_file(image_path)
face_locs = face_recognition.face_locations(image)
landmarks_list = face_recognition.face_landmarks(image)

if not face_locs or not landmarks_list:
    raise RuntimeError("얼굴을 찾을 수 없습니다.")

# use first face
top, right, bottom, left = face_locs[0]
fw, fh = right - left, bottom - top

lm = landmarks_list[0]
left_eye_pts   = lm["left_eye"]
right_eye_pts  = lm["right_eye"]
nose_bridge    = lm["nose_bridge"]
nose_tip_pts   = lm["nose_tip"]
top_lip_pts    = lm["top_lip"]
bottom_lip_pts = lm["bottom_lip"]
chin_pts       = lm["chin"]

# centroids & key points
left_eye_cent  = centroid(left_eye_pts)
right_eye_cent = centroid(right_eye_pts)
nose_tip_cent  = centroid(nose_tip_pts)
mouth_cent     = centroid(top_lip_pts + bottom_lip_pts)
chin_bottom    = chin_pts[len(chin_pts)//2]

# compute distances
metrics = {}

metrics["face_width"]  = fw
metrics["face_height"] = fh

metrics["eye_distance"]    = euclidean_dist(left_eye_cent, right_eye_cent)
metrics["left_eye_width"]  = euclidean_dist(left_eye_pts[0], left_eye_pts[-1])
metrics["right_eye_width"] = euclidean_dist(right_eye_pts[0], right_eye_pts[-1])
metrics["left_eye_height"] = euclidean_dist(left_eye_pts[1], left_eye_pts[5])
metrics["right_eye_height"]= euclidean_dist(right_eye_pts[1], right_eye_pts[5])

metrics["nose_length"]     = euclidean_dist(nose_bridge[0], nose_tip_pts[-1])
metrics["nose_width"]      = euclidean_dist(nose_tip_pts[0], nose_tip_pts[-1])

metrics["mouth_width"]     = euclidean_dist(top_lip_pts[0], top_lip_pts[6])
metrics["mouth_height"]    = euclidean_dist(centroid(top_lip_pts), centroid(bottom_lip_pts))

metrics["eye_to_mouth"]    = euclidean_dist(((left_eye_cent[0]+right_eye_cent[0])/2, (left_eye_cent[1]+right_eye_cent[1])/2), mouth_cent)
metrics["eye_left_to_chin"]= euclidean_dist(left_eye_cent, chin_bottom)
metrics["eye_right_to_chin"]= euclidean_dist(right_eye_cent, chin_bottom)
metrics["nose_to_mouth"]   = euclidean_dist(nose_tip_cent, mouth_cent)
metrics["nose_to_chin"]    = euclidean_dist(nose_tip_cent, chin_bottom)

metrics["jaw_width"]       = euclidean_dist(chin_pts[0], chin_pts[-1])
metrics["jaw_length"]      = sum(euclidean_dist(chin_pts[i], chin_pts[i+1]) for i in range(len(chin_pts)-1))

# ratios
ratios = { k + "_ratio_width":  metrics[k] / fw for k in [
    "eye_distance", "left_eye_width", "right_eye_width",
    "nose_length", "nose_width", "mouth_width", "jaw_width"
] }
ratios.update({ k + "_ratio_height": metrics[k] / fh for k in [
    "left_eye_height", "right_eye_height", "mouth_height"
] })
ratios["eye_to_mouth_ratio"]   = metrics["eye_to_mouth"] / fh
ratios["eye_left_to_chin_ratio"]= metrics["eye_left_to_chin"] / fh
ratios["eye_right_to_chin_ratio"]= metrics["eye_right_to_chin"] / fh
ratios["nose_to_mouth_ratio"]  = metrics["nose_to_mouth"] / fh
ratios["nose_to_chin_ratio"]   = metrics["nose_to_chin"] / fh
ratios["jaw_length_ratio"]     = metrics["jaw_length"] / fw

# angles
angles = {}
angles["eye_axis_angle"]   = slope_angle(left_eye_cent, right_eye_cent)
angles["nose_axis_angle"]  = slope_angle(nose_bridge[0], nose_tip_pts[-1])
angles["mouth_axis_angle"] = slope_angle(top_lip_pts[0], top_lip_pts[6])
angles["jaw_axis_angle"]   = slope_angle(chin_pts[0], chin_pts[-1])

# jaw angle between left & right jaw from center
jaw_center = centroid([chin_pts[0], chin_pts[-1]])
v1 = (chin_pts[0][0] - jaw_center[0], chin_pts[0][1] - jaw_center[1])
v2 = (chin_pts[-1][0] - jaw_center[0], chin_pts[-1][1] - jaw_center[1])
angles["jaw_angle"]        = vector_angle(v1, v2)

# nose angle at tip
v1 = (nose_tip_pts[0][0] - nose_tip_cent[0], nose_tip_pts[0][1] - nose_tip_cent[1])
v2 = (nose_tip_pts[-1][0] - nose_tip_cent[0], nose_tip_pts[-1][1] - nose_tip_cent[1])
angles["nose_angle"]       = vector_angle(v1, v2)

# areas
areas = {}
areas["left_eye_area"]   = polygon_area(left_eye_pts)
areas["right_eye_area"]  = polygon_area(right_eye_pts)
areas["mouth_area"]      = polygon_area(top_lip_pts + bottom_lip_pts[::-1])
areas["nose_area"]       = polygon_area(nose_bridge + nose_tip_pts[::-1])
# approximate face area as bounding box
areas["face_area"]       = fw * fh
areas["half_face_area"]  = areas["face_area"] / 2

# symmetry
symmetry = {}
mid_x = left + fw/2
symmetry["eye_symmetry"]   = (abs(left_eye_cent[0] - mid_x) + abs(right_eye_cent[0] - mid_x)) / fw
symmetry["mouth_symmetry"] = abs(top_lip_pts[0][0] + top_lip_pts[6][0] - 2*mid_x) / fw
symmetry["nose_symmetry"]  = abs(nose_bridge[0][0] + nose_bridge[-1][0] - 2*mid_x) / fw

# advanced composite
advanced = {}
advanced["golden_ratio_diff"] = abs((metrics["eye_distance"] / metrics["mouth_width"]) - 1.618)
average_zone = (metrics["eye_distance"] + metrics["nose_width"]) / 2
advanced["t_zone_index"]      = average_zone / metrics["eye_distance"]
# face region ratios: top→nose_bridge, nose_bridge→nose_tip, nose_tip→chin
advanced["face_region_top"]   = (nose_bridge[0][1] - top) / fh
advanced["face_region_mid"]   = (nose_tip_cent[1] - nose_bridge[0][1]) / fh
advanced["face_region_bot"]   = (chin_bottom[1] - nose_tip_cent[1]) / fh

# collate all
face_metrics = {
    "distances": metrics,
    "ratios":    ratios,
    "angles":    angles,
    "areas":     areas,
    "symmetry":  symmetry,
    "advanced":  advanced
}

import json
print(json.dumps(face_metrics, indent=2))
