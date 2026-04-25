import subprocess
import sys

def run_training():
    # 定义训练命令
    command = [
        "python", "yolov5/train.py",
        "--img", "640",  # 图像大小
        "--batch", "4",  # 批处理大小
        "--epochs", "500",  # 训练轮数
        "--data", "throat_data.yaml",  # 数据配置文件路径
        "--cfg", "yolov5/models/yolov5s.yaml",  # 模型配置文件路径
        "--weights", "yolov5s.pt",  # 预训练权重
        "--name", "throat_train",  # 运行名称
        "--workers", "2"  # 增加工作线程数
    ]

    try:
        print("Starting training...")
        result = subprocess.run(command, check=True)
        print("Training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_training()



