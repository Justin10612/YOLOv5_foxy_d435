import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import os
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
from std_msgs.msg import Bool

class yolov5_ros(Node):


    count = 0
    d_list = []
    distance = 0

    def __init__(self):
        # Node Initialize
        super().__init__('human_detector')
        # Publisher Initialize
        self.publisher_ = self.create_publisher(Vector3, 'human_pose', 10)
        self.target_status_pub_ = self.create_publisher(Bool, 'target_status', 10)
        # Select Model
        model_name = 'best_c_v8.pt'
        model_path = '/home/sss0301/ros2_ws/src/detection/weights/'
        # path_ = os.path.join(self.model_name, self.model_path)
        path_ = model_path + model_name 
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_)
        # self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # Set Confidence
        self.model.conf = 0.5
    
# Clamp function
    def clamp(self, n, smallest, largest):
        return max(smallest, min(n, largest))

    # The func that can let U know the distance
    def get_mid_pos_A(self, org_img, box, depth_data):
        distance_list = []
        depth_heigh = 250
        bias = 10
        target_x = int(self.clamp((box[0] + box[2])//2, 0, 1280))    # The center-x of the bounding box
        # Get the smaple point
        for i in range(20):
            target_y = int(self.clamp(depth_heigh+bias, 0, 720))
            # Let you know where the smaple point is.
            cv2.circle(org_img, (target_x, target_y), 8, (255,255,255), -1)
            distance_list.append(depth_data[target_y, target_x])
            bias +=10
        return np.mean(distance_list)
    
    def dectshow(self, org_img, boxs, depth_data):
        pose_msg = Vector3()
        status_msg = Bool()

        # The func that can let U know the distance
        # def get_mid_pos_B(box, depth_data):
        #     distance_list = []
        #     depth_heigh = 250
        #     bias = 10
        #     target_x = int(clamp((box[0] + box[2])//2, 0, 1280))    # The center-x of the bounding box
        #     target_y = int(clamp(depth_heigh), 0, 720)
        #     # Get the smaple point
        #     for i in range(5):
        #         # Let you know where the smaple point is.
        #         distance_list.append(depth_data[
        #             random.randint(target_y-bias, target_y+bias), 
        #             random.randint(target_x-bias, target_x+bias)])
        #         cv2.circle(org_img, (target_x, target_y), 8, (255,255,255), 2)
        #     return np.mean(distance_list)
        
        if len(boxs)==0:
            # Publish Target Pose
            pose_msg.x = 0.0  # x
            pose_msg.y = 0.0  # depth
            self.publisher_.publish(pose_msg)
            # Publish Target Status
            status_msg.data = False
            self.target_status_pub_.publish(status_msg)
            self.get_logger().info('Target Lost')
        
        # Gvie Every Box Distance_Value
        for box in boxs:
            # Drawing the Bounding Box
            cv2.rectangle(org_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # Calculate the Distance

            for i in range(4):
                self.d_list.append(self.get_mid_pos_A(org_img, box, depth_data))
            self.d_list.sort()
            self.distance = self.d_list[2]
            # self.get_logger.info('Dis %f' % self.distance)
            self.d_list = []
            cv2.putText(org_img, 
                        str(float(box[4]))[:4],
                        (107, 77), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (3, 255, 65), 2)
            # THIS CAN ONLY DEAL WITH SINGLE OBJET
            # Show Name and Distance
            cv2.putText(org_img, 
                        "Vest " + str(float(self.distance) / 1000)[:4] + 'm',
                        (int(box[0]), int(box[3])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, (255, 255, 255), 2)
            ################ Publish Data ################
            # Publish Target Pose
            pose_msg.x = (box[0] + box[2])//2  # x
            pose_msg.y = float(self.distance)/10              # depth
            self.publisher_.publish(pose_msg)
            # Publish Target Status
            status_msg.data = True
            self.target_status_pub_.publish(status_msg)
            self.get_logger().info('Locked on target')
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
                # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=1), cv2.COLORMAP_JET)
                # # Show images
                # cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('Depth', depth_colormap)

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