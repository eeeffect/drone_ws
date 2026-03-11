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
    
    # We can also check if the parameter matcher_type exists and defaults to orb
    assert node.get_parameter('matcher_type').get_parameter_value().string_value == 'orb'
    
    node.destroy_node()
    rclpy.shutdown()
