import cv2
import numpy as np
import random
import pyrealsense2 as rs

# 设置保存图像的文件夹路径
output_folder = 'image_saves/'

# 初始化计数器和帧间隔
frame_count = 0
frame_interval = 15  # 每隔30帧保存一次图像
picture_num = 0

# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 配置所需的流

pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    if color_frame:
        # 将RealSense的帧数据转换为NumPy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 将NumPy数组转换为OpenCV的Mat对象
        cv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

        frame_count += 1

        # 保存图像
        if frame_count % frame_interval == 0:
            image_filename = f"{output_folder}frame_{random.randint(1,10000)}.jpg"
            cv2.imwrite(image_filename, cv_image)
            print(f"Saved {image_filename}")
            picture_num += 1

        cv2.putText(cv_image, str(picture_num), (107, 77), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 1)
        # 在这里，cv_image 是一个OpenCV的Mat对象，您可以对其进行进一步的处理或保存
        cv2.imshow("RealSense Image", cv_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 关闭相机和窗口
pipeline.stop()
cv2.destroyAllWindows()
