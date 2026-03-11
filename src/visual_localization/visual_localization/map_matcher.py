import cv2
import numpy as np
import logging

class BaseMatcher:
    """Abstract base class for all Map Matchers"""
    def __init__(self, map_image_path):
        self.map_img = cv2.imread(map_image_path, cv2.IMREAD_GRAYSCALE)
        if self.map_img is None:
            raise ValueError(f"Could not load map image from {map_image_path}")

    def match(self, frame_img):
        """Must return a dictionary with success, x, y, yaw, inliers"""
        raise NotImplementedError
        
    def _compute_pose_from_homography(self, M, mask, frame_shape):
        """Helper to convert Homography matrix back to Pose"""
        inliers = int(np.sum(mask)) if mask is not None else 0
        MIN_MATCH_COUNT = 15
        
        if M is not None and inliers > MIN_MATCH_COUNT * 0.8:
            h, w = frame_shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            center_x = np.mean(dst[:, 0, 0])
            center_y = np.mean(dst[:, 0, 1])
            theta = np.arctan2(M[1, 0], M[0, 0])
            
            return {
                'success': True,
                'x': float(center_x),
                'y': float(center_y),
                'yaw': float(theta),
                'inliers': inliers
            }
        return {'success': False, 'inliers': inliers}


class ORBMatcher(BaseMatcher):
    def __init__(self, map_image_path):
        super().__init__(map_image_path)
        self.orb = cv2.ORB_create(nfeatures=5000)
        self.map_kp, self.map_des = self.orb.detectAndCompute(self.map_img, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match(self, frame_img):
        if len(frame_img.shape) == 3:
            frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_img
            
        frame_kp, frame_des = self.orb.detectAndCompute(frame_gray, None)
        if frame_des is None or len(frame_des) < 10:
            return {'success': False, 'inliers': 0}
            
        try:
            matches = self.matcher.knnMatch(frame_des, self.map_des, k=2)
        except Exception:
            return {'success': False, 'inliers': 0}
            
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        MIN_MATCH_COUNT = 15
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([frame_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.map_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return self._compute_pose_from_homography(M, mask, frame_gray.shape)
            
        return {'success': False, 'inliers': len(good_matches)}


class KorniaMatcher(BaseMatcher):
    """Deep learning matcher using Kornia (LoFTR or SuperPoint/LightGlue)"""
    def __init__(self, map_image_path, model_type='loftr'):
        super().__init__(map_image_path)
        self.model_type = model_type
        
        try:
            import torch
            import kornia as K
            from kornia.feature import LoFTR, LightGlueMatcher
        except ImportError:
            raise ImportError("PyTorch or Kornia not installed. Cannot use Deep matchers. Run: pip install torch kornia")
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.getLogger().info(f"Initializing {model_type} on {self.device}")
        
        # Load the map into a PyTorch Tensor (batch_size=1, channels=1, H, W)
        self.map_tensor = K.image_to_tensor(self.map_img, False).float() / 255.0
        self.map_tensor = self.map_tensor.to(self.device)
        
        # Initialize selected model
        if model_type == 'loftr':
            self.matcher = LoFTR(pretrained='outdoor').to(self.device)
        elif model_type == 'superpoint':
            self.matcher = LightGlueMatcher('superpoint').to(self.device)
        else:
            raise ValueError(f"Unknown deep matcher: {model_type}")
            
        self.matcher.eval()

    def match(self, frame_img):
        import torch
        import kornia as K
        
        if len(frame_img.shape) == 3:
            frame_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame_img
            
        # Convert frame to PyTorch tensor
        frame_tensor = K.image_to_tensor(frame_gray, False).float() / 255.0
        frame_tensor = frame_tensor.to(self.device)
        
        input_dict = {
            "image0": frame_tensor, # Query image (Drone camera)
            "image1": self.map_tensor # Reference image (Orthophoto)
        }
        
        with torch.inference_mode():
            if self.model_type == 'loftr':
                correspondences = self.matcher(input_dict)
                mkpts0 = correspondences['keypoints0'].cpu().numpy()
                mkpts1 = correspondences['keypoints1'].cpu().numpy()
            elif self.model_type == 'superpoint':
                # superpoint structure in kornia:
                # Returns keypoints and match indices
                # Note: Exact api may require running the local feature extractor first depending on kornia version, 
                # but let's use the provided matcher wrapper.
                try:
                     outputs = self.matcher(input_dict)
                     mkpts0 = outputs['keypoints0'].cpu().numpy()
                     mkpts1 = outputs['keypoints1'].cpu().numpy()
                except Exception as e:
                     logging.getLogger().error(f"Superpoint match failed: {e}")
                     return {'success': False, 'inliers': 0}
            
        if len(mkpts0) < 15:
            return {'success': False, 'inliers': len(mkpts0)}
            
        # Homography from deep points
        src_pts = np.float32(mkpts0).reshape(-1, 1, 2)
        dst_pts = np.float32(mkpts1).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return self._compute_pose_from_homography(M, mask, frame_gray.shape)

def create_matcher(matcher_type, map_image_path):
    """Factory to create the right matcher based on user config"""
    if matcher_type.lower() == 'orb':
        return ORBMatcher(map_image_path)
    elif matcher_type.lower() in ['loftr', 'superpoint']:
        return KorniaMatcher(map_image_path, model_type=matcher_type.lower())
    else:
        raise ValueError(f"Matcher {matcher_type} not supported.")
