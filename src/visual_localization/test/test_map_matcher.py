import os
import cv2
import numpy as np
import pytest
import tempfile
from visual_localization.map_matcher import create_matcher, ORBMatcher

@pytest.fixture
def synthetic_map():
    # Create a 1000x1000 map
    img = np.zeros((1000, 1000), dtype=np.uint8)
    
    # Create a 1000x1000 map with random noise and blur to create natural-looking features
    np.random.seed(42)  # Deterministic tests
    
    # Generate random blobs
    img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_LANCZOS4)
    
    # Add some high contrast shapes
    for _ in range(50):
        x1, y1 = np.random.randint(0, 900, 2)
        x2, y2 = x1 + np.random.randint(50, 150, 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(img, (x1+5, y1+5), (x2-5, y2-5), (0, 0, 0), -1)

    # Save to a temporary file
    fd, path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    cv2.imwrite(path, img)
    
    yield path, img
    
    # Cleanup
    os.remove(path)

def test_map_matcher_exact_subimage(synthetic_map):
    map_path, map_img = synthetic_map
    matcher = create_matcher('orb', map_path)
    
    # Crop a 200x200 region from the center (400 to 600)
    # The center of this region in the map is at (500, 500)
    frame = map_img[400:600, 400:600].copy()
    
    result = matcher.match(frame)
    
    assert result['success'] is True
    assert result['inliers'] > 10
    
    # Center should be around 500, 500
    assert abs(result['x'] - 500) < 5.0
    assert abs(result['y'] - 500) < 5.0
    
    # Yaw should be ~0
    assert abs(result['yaw']) < 0.1

def test_map_matcher_rotated_subimage(synthetic_map):
    map_path, map_img = synthetic_map
    matcher = create_matcher('orb', map_path)
    
    # Crop a region
    frame = map_img[300:600, 300:600].copy()  # 300x300 center at 450, 450
    
    # Rotate the frame by 90 degrees clockwise
    # center is 150, 150 in the frame
    M = cv2.getRotationMatrix2D((150, 150), -90, 1.0)
    rotated_frame = cv2.warpAffine(frame, M, (300, 300))
    
    result = matcher.match(rotated_frame)
    
    assert result['success'] is True
    
    # Center should still be around (450, 450)
    assert abs(result['x'] - 450) < 10.0
    assert abs(result['y'] - 450) < 10.0
    
    # Yaw should be roughly -pi/2 or pi/2 depending on conventions
    # ORB homography rotation estimation accuracy can vary, so we check large bounds
    import math
    expected_yaw = -math.pi / 2
    
    # Normalize angles to handle periodicity
    diff = (result['yaw'] - expected_yaw + math.pi) % (2 * math.pi) - math.pi
    assert abs(diff) < 0.2

def test_map_matcher_no_match(synthetic_map):
    map_path, _ = synthetic_map
    matcher = create_matcher('orb', map_path)
    
    # Blank frame, should fail
    blank_frame = np.zeros((200, 200), dtype=np.uint8)
    
    result = matcher.match(blank_frame)
    
    assert result['success'] is False
    assert result['inliers'] < 10
