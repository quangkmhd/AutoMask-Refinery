import cv2
import numpy as np

def compute_mask_area(points):
    """Compute area of a polygon defined by points."""
    return cv2.contourArea(np.array(points, dtype=np.float32))

def compute_bbox_area(bbox):
    """Compute area of [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1)) * max(0, (y2 - y1))

def compute_compactness(points):
    """compactness = perimeter^2 / area"""
    area = compute_mask_area(points)
    if area == 0:
        return 0
    perimeter = cv2.arcLength(np.array(points, dtype=np.float32), True)
    return (perimeter ** 2) / area

def compute_aspect_ratio(points):
    """aspect_ratio = mask_width / mask_height"""
    if not points:
        return 0
    points = np.array(points, dtype=np.float32)
    x, y, w, h = cv2.boundingRect(points)
    if h == 0:
        return 0
    return w / h

def compute_solidity(points):
    """solidity = area / convex_hull_area"""
    area = compute_mask_area(points)
    if area == 0:
        return 0
    points_arr = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points_arr)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0
    return area / hull_area

def compute_hu_moments(points):
    """7 Hu moments from points."""
    moments = cv2.moments(np.array(points, dtype=np.float32))
    hu = cv2.HuMoments(moments).flatten()
    # Log transform to make them more manageable
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)
    return hu_log
