import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
import numpy as np

from visual_localization.map_matcher import MapMatcher

class VisualLocalizationNode(Node):
    def __init__(self):
        super().__init__('visual_localization_node')
        
        # Declare parameters
        self.declare_parameter('map_path', '')
        map_path = self.get_parameter('map_path').get_parameter_value().string_value
        
        if not map_path:
            self.get_logger().error("Map path parameter 'map_path' not set! Cannot initialize MapMatcher.")
            self.matcher = None
        else:
            try:
                self.matcher = MapMatcher(map_path)
                self.get_logger().info(f"Successfully loaded map from {map_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize MapMatcher: {e}")
                self.matcher = None
            
        self.bridge = CvBridge()
        
        # Subscribe to camera images
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publish PoseWithCovarianceStamped
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/visual_pose',
            10
        )
        self.get_logger().info("Visual Localization Node started.")
        
    def image_callback(self, msg):
        if self.matcher is None:
            return
            
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
            
        result = self.matcher.match(cv_image)
        
        if result['success']:
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header = msg.header
            pose_msg.header.frame_id = 'map'
            
            # Map pixel coordinates to map frame coordinates
            pose_msg.pose.pose.position.x = result['x']
            pose_msg.pose.pose.position.y = result['y']
            pose_msg.pose.pose.position.z = 0.0 
            
            # 2D Yaw to Quaternion
            yaw = result['yaw']
            pose_msg.pose.pose.orientation.x = 0.0
            pose_msg.pose.pose.orientation.y = 0.0
            pose_msg.pose.pose.orientation.z = float(np.sin(yaw / 2.0))
            pose_msg.pose.pose.orientation.w = float(np.cos(yaw / 2.0))
            
            # Dynamic covariance based on inliers
            inliers = result['inliers']
            base_cov = 0.5
            if inliers > 20: 
                cov = max(0.05, base_cov - (inliers - 20) * 0.015)
            else:
                cov = base_cov
                
            # 6x6 covariance matrix (row-major)
            cov_matrix = [0.0] * 36
            cov_matrix[0]  = cov  # x
            cov_matrix[7]  = cov  # y
            cov_matrix[14] = 999.0 # z (unknown)
            cov_matrix[21] = 999.0 # roll (unknown)
            cov_matrix[28] = 999.0 # pitch (unknown)
            cov_matrix[35] = cov  # yaw
            
            pose_msg.pose.covariance = cov_matrix
            
            self.pose_pub.publish(pose_msg)
            self.get_logger().debug(f"Published pose X:{result['x']:.2f} Y:{result['y']:.2f} Yaw:{result['yaw']:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = VisualLocalizationNode()
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
