import os
import cv2
from tkinter import Tk, Label, Button, messagebox
from PIL import Image, ImageTk
import pyrealsense2 as rs
import numpy as np


class PhotoCaptureApp:
    # 图片保存路径配置 - 修改这里可以更改保存的文件夹名称
    IMAGE_SAVE_FOLDER = 'throat_1'  # 可以改为: 'capture_image', 'food_dataset', 'meal_photos' 等

    def __init__(self, master):
        self.master = master
        master.title("拍照并储存图片")

        # 初始化RealSense管道
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        self.capture_mode = False
        self.image_count = self.get_next_image_number()  # 获取下一个图片编号
        self.scale_factor = 1.0  # 初始化缩放因子

        self.label = Label(master, text="请开始拍摄")
        self.label.pack(pady=10)

        self.capture_button = Button(master, text="开始拍摄", command=self.toggle_capture)
        self.capture_button.pack(pady=10)

        self.stop_capture_button = Button(master, text="停止拍摄", command=self.stop_capture, state='disabled')
        self.stop_capture_button.pack(pady=10)

        self.photo_button = Button(master, text="拍照", command=self.capture_photo, state='disabled')
        self.photo_button.pack(pady=10)

        self.image_label = Label(master)
        self.image_label.pack()

        self.update_camera_feed()

    def update_camera_feed(self):
        """更新相机画面"""
        if self.capture_mode:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                return

            frame = np.asanyarray(color_frame.get_data())

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 获取窗口宽度和高度
            window_width = self.master.winfo_screenwidth() * 0.8
            window_height = self.master.winfo_screenheight() * 0.8

            # 计算缩放因子
            original_height, original_width, _ = frame_rgb.shape
            aspect_ratio = original_width / original_height
            if original_width > window_width or original_height > window_height:
                if aspect_ratio > 1:
                    self.scale_factor = window_width / original_width
                else:
                    self.scale_factor = window_height / original_height

            new_width = int(original_width * self.scale_factor)
            new_height = int(original_height * self.scale_factor)
            resized_frame = cv2.resize(frame_rgb, (new_width, new_height))

            img = Image.fromarray(resized_frame)
            img_tk = ImageTk.PhotoImage(image=img)
            self.image_label.img_tk = img_tk
            self.image_label.config(image=img_tk)

        self.master.after(30, self.update_camera_feed)

    def toggle_capture(self):
        """切换拍摄模式"""
        if not self.capture_mode:
            self.capture_mode = True
            self.capture_button.config(text="停止拍摄")
            self.stop_capture_button.config(state='normal')
            self.photo_button.config(state='normal')
        else:
            self.capture_mode = False
            self.capture_button.config(text="开始拍摄")
            self.stop_capture_button.config(state='disabled')
            self.photo_button.config(state='disabled')

    def stop_capture(self):
        """停止拍摄"""
        self.capture_mode = False
        self.capture_button.config(text="开始拍摄")
        self.stop_capture_button.config(state='disabled')
        self.photo_button.config(state='disabled')

    def capture_photo(self):
        """捕获照片并保存"""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            return

        frame = np.asanyarray(color_frame.get_data())

        save_dir = os.path.join(os.path.dirname(__file__), self.IMAGE_SAVE_FOLDER)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filename = f"image_{self.image_count}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        self.image_count += 1
        print(f"Captured {filepath}")
        #messagebox.showinfo("成功", f"图片已保存为 {filename}")

    def get_next_image_number(self):
        """获取下一个图片编号"""
        read_dir = os.path.join(os.path.dirname(__file__), self.IMAGE_SAVE_FOLDER)
        if not os.path.exists(read_dir):
            os.makedirs(read_dir)

        files = [f for f in os.listdir(read_dir) if f.startswith('image_') and f.endswith('.jpg')]
        numbers = [int(f.split('_')[1].split('.')[0]) for f in files]
        return max(numbers) + 1 if numbers else 0

    def __del__(self):
        # 停止管道
        self.pipeline.stop()


if __name__ == "__main__":
    root = Tk()
    app = PhotoCaptureApp(root)

    # 启动主循环
    root.mainloop()
