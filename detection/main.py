import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3

class yolov5_ros(Node):

    def __init__(self):
        super().__init__('human_detector')
        self.publisher_ = self.create_publisher(Vector3, 'human_pos', 10)
        # 
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
        self.model.conf = 0.5
        
    def dectshow(self, org_img, boxs, depth_data):
        def get_mid_pos(box, depth_data):
            distance_list = []
            depth_heigh = 400
            bias = 10
            mid_x = (box[0] + box[2])//2    # The center-x of the bounding box
            distance_list.append(depth_data[mid_x-bias, depth_heigh])
            distance_list.append(depth_data[mid_x, depth_heigh])
            distance_list.append(depth_data[mid_x+bias, depth_heigh])
            return np.mean(distance_list)
        
        img = org_img.copy()
        for box in boxs:
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            #dist = get_mid_pos(box, depth_data)
            dist = 6665.1
            cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                        (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('dec_img', img)
    
    def run_detection(self):
        # Configure depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
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

                results = self.model(color_image)
                boxs= results.pandas().xyxy[0].values
                #boxs = np.load('temp.npy',allow_pickle=True)
                self.dectshow(color_image, boxs, depth_image)

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # Stack both images horizontally
                images = np.hstack((color_image, depth_colormap))
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)
                # Publish Data
                msg = Vector3()
                msg.x = 666.1  # angle
                msg.y = 541.21 # depth
                self.publisher_.publish(msg)
                self.get_logger().info('Human.angle "%f", Human.depth "%f"' % (msg.x, msg.y))
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