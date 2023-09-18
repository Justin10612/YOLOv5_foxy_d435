import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import os

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3

class yolov5_ros(Node):

    def __init__(self):
        super().__init__('human_detector')
        self.publisher_ = self.create_publisher(Vector3, 'human_pos', 10)
        # Select Model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/sss0301/ros2_ws/src/detection/models/bestv8.pt', force_reload=True)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.model.conf = 0.5
        
    def dectshow(self, org_img, boxs, depth_data):
        def clamp(n, smallest, largest):
            return max(smallest, min(n, largest))
        # The func that can lat U know the distance
        def get_mid_pos(box, depth_data):
            distance_list = []
            depth_heigh = 250
            bias = 10
            target_x = int(clamp((box[0] + box[2])//2, 0, 1280))    # The center-x of the bounding box
            # Get the smaple point
            for i in range(5):
                target_y = int(clamp(depth_heigh+bias, 0, 720))
                cv2.circle(org_img, (target_x, target_y), 8, (255,255,255), 2)
                distance_list.append(depth_data[target_y, target_x])
                bias +=30
                # self.get_logger().info('Human.angle "%f"' % (target_y))
            # Let you know where the smaple point is.
            # cv2.circle(org_img, (int(target_x), int(depth_heigh)), 8, (255,255,255), 2)
            return np.mean(distance_list)
        
        # Gvie Every Box Distance_Value
        for box in boxs:
            # Drawing the Bounding Box
            cv2.rectangle(org_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # Calculate the Distance
            distance = get_mid_pos(box, depth_data)
            # THIS CAN ONLY DEAL WITH SINGLE OBJET
            # Show Name and Distance
            cv2.putText(org_img, 
                        "Target" + str(float(distance) / 1000)[:4] + 'm',
                        (int(box[0]), int(box[3])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (255, 255, 255), 1)
            # Publish Data
            msg = Vector3()
            msg.x = (box[0] + box[2])//2  # x
            msg.y = distance              # depth
            self.publisher_.publish(msg)
            self.get_logger().info('Human.angle "%.2f", Human.depth "%.2f"' % (msg.x, msg.y))
        cv2.imshow('dec_img', org_img)
    
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
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                # print(np.shape(depth_image))

                results = self.model(color_image)
                boxs= results.pandas().xyxy[0].values
                # Main detection
                self.dectshow(color_image, boxs, depth_image)

                # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # # Stack both images horizontally
                # images = np.hstack((color_image, depth_colormap))
                # # Show images
                # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('RealSense', images)

                # Press esc or 'q' to close the image window
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
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