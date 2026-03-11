import cv2
import numpy as np

class MapMatcher:
    def __init__(self, map_image_path):
        """
        Initialize the MapMatcher with a reference map image.
        Uses ORB for fast feature extraction currently (suitable for CPU/basic prototyping).
        """
        self.map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if self.map_img is None:
            raise ValueError(f"Could not load map image from {map_image_path}")
            
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=5000)
        
        # Detect and compute keypoints and descriptors for the map
        self.map_kp, self.map_des = self.orb.detectAndCompute(self.map_img, None)
        
        # Flann-based matcher with LSH for ORB
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, frame_img):
        """
        Match the incoming frame with the map to find its position.
        
        Args:
            frame_img: A numpy array representing the image from the drone camera.
            
        Returns:
            dict containing:
                'success': boolean indicating if a good match was found
                'x': x coordinate in map pixels (center of frame)
                'y': y coordinate in map pixels
                'yaw': rotation in radians
                'inliers': number of good matches found (can be used for covariance)
        """
        if len(frame_img.shape) == 3:
            frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_img
            
        frame_kp, frame_des = self.orb.detectAndCompute(frame_gray, None)
        
        if frame_des is None or len(frame_des) < 10:
            return {'success': False, 'inliers': 0}
            
        # KNN Match
        try:
            matches = self.matcher.knnMatch(frame_des, self.map_des, k=2)
        except Exception:
            return {'success': False, 'inliers': 0}
            
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        MIN_MATCH_COUNT = 15
        if len(good_matches) > MIN_MATCH_COUNT:
            # Get coordinates of matched keypoints
            src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.map_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Find Homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(np.sum(mask)) if mask is not None else 0
            
            if M is not None and inliers > MIN_MATCH_COUNT * 0.8:
                h, w = frame_gray.shape
                # Coordinates of the frame corners [x, y]
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # Center point
                center_x = np.mean(dst[:, 0, 0])
                center_y = np.mean(dst[:, 0, 1])
                
                # Approximate Yaw
                theta = np.arctan2(M[1, 0], M[0, 0])
                
                return {
                    'success': True,
                    'x': float(center_x),
                    'y': float(center_y),
                    'yaw': float(theta),
                    'inliers': inliers
                }
                
        return {'success': False, 'inliers': len(good_matches)}
