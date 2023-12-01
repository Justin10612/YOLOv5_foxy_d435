import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import os
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool
from ament_index_python.packages import get_package_share_directory

class yolov5_ros(Node):

    d_list = []
    distance = 0
    target_state = True
    MODEL_NAME= 'best_c_v11.pt'
    MODEL_PATH = '/home/yang/ros2_ws/src/detection/weights'

    def __init__(self):
        # Node Initialize
        super().__init__('human_detector')
        # Publisher Initialize
        self.publisher_ = self.create_publisher(Vector3, 'human_pose', 10)
        self.target_status_pub_ = self.create_publisher(Bool, 'target_status', 10)
        # Select Model
        path_ = os.path.join(self.MODEL_PATH, self.MODEL_NAME)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s6')
        # Set Confidence
        self.model.conf = 0.5

    def dectshow(self, boxs, depth_data):
        pose_msg = Vector3()
        status_msg = Bool()
        if len(boxs)==0:
            # Publish Target Pose
            pose_msg.x = 0.0  # x
            pose_msg.y = 0.0  # depth
            pose_msg.z = 0.0
            # Update Target Status
            status_msg.data = False
            
        # Gvie Every Box Distance_Value
        for box in boxs:
            # Midan Filter
            for i in range(3):
                self.d_list.append(self.get_distance_x_type(box, depth_data))
            self.d_list.sort()
            self.distance = self.d_list[1]
            self.d_list = []
            # Update Target Pose
            pose_msg.x = (box[0] + box[2])//2  # x
            pose_msg.y = self.distance  # depth mili-meters
            pose_msg.z = 1.0    # Target Status
            # Update Target Status
            status_msg.data = True
        
        # Log Data
        if status_msg.data != self.target_state:
            if status_msg.data:
                self.get_logger().info('Locked on target')
            else:
                self.get_logger().info('Target Lost')
        self.target_state  = status_msg.data

        # Publish Target Pose
        self.publisher_.publish(pose_msg)
        self.target_status_pub_.publish(status_msg)

    # Clamp function
    def clamp(self, n, smallest, largest):
        return int(max(smallest, min(n, largest)))

    # The func that can let U know the distance
    def get_distance_x_type(self, box, depth_data):
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        bias = 10
        distance_list = []
        # The center-x of the bounding box
        target_x = (x0 + x1)//2
        # The center-y of the bounding box
        target_y = (y0 + y1)//2
        # Get the smaple point
        for i in range(20):
            y_plus = min(target_y+bias, y1-1)
            y_minus = max(target_y-bias, y0+1)
            # Let you know where the smaple point is.
            distance_list.append(depth_data[y_plus, target_x])
            distance_list.append(depth_data[y_minus, target_x])
            bias +=10
        return np.mean(distance_list)
    
    def run_detection(self):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        # Start streaming
        pipeline.start(config)
        try:
            while True:
                # start_time = time.time()
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                # Convert images to numpy arrays
                filled_depth_frame = rs.hole_filling_filter().process(depth_frame)
                depth_image = np.asanyarray(filled_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                # Detect Result
                results = self.model(color_image)
                boxs= results.pandas().xyxy[0].values
                # Main detection
                self.dectshow( boxs, depth_image)

                # Press esc or 'q' to close the image window
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # fps = 1/elapsed_time
                # self.get_logger().info('FPS: %.2f' % fps)
        finally:
            # Stop streaming
            pipeline.stop()

def detect_main(args=None):
    rclpy.init(args=args)
    human_detector = yolov5_ros()
    human_detector.run_detection()
    rclpy.spin(human_detector)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    human_detector.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    detect_main()