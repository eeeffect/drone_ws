import pytest
import rclpy
from visual_localization.localization_node import VisualLocalizationNode

def test_node_init():
    try:
        rclpy.init()
    except Exception:
        pass # Already initialized
        
    node = VisualLocalizationNode()
    
    assert node.get_name() == 'visual_localization_node'
    # By default parameter map_path is empty, so matcher should be None
    assert getattr(node, 'matcher', None) is None
    
    node.destroy_node()
    rclpy.shutdown()
