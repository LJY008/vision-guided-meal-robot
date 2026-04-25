import os
import shutil
from pathlib import Path
import random

# 路径配置 - 修改这里可以更改输入和输出文件夹名称
ORIGINAL_IMAGES_FOLDER = 'throat_1'     # 原始图像文件夹
ORIGINAL_LABELS_FOLDER = 'throat_label1'       # 原始标签文件夹
DATASET_OUTPUT_FOLDER = 'throat_dataset1'      # 输出数据集文件夹

# 假设你的图像和标注文件在 'original_images/' 和 'original_labels/' 中
original_images_dir = os.path.join(os.path.dirname(__file__), ORIGINAL_IMAGES_FOLDER)
original_labels_dir = os.path.join(os.path.dirname(__file__), ORIGINAL_LABELS_FOLDER)

# 定义训练集和验证集的比例（例如80%训练，20%验证）
train_ratio = 0.8

# 获取所有图像文件名
image_files = list(Path(original_images_dir).glob('*.jpg'))  # 根据实际情况调整扩展名

if not image_files:
    print("没有找到任何图像文件，请检查路径是否正确。")
    exit()

# 获取目标文件夹中已存在的图像文件名
existing_train_images = set(Path(os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/images/train')).glob('*.jpg'))
existing_val_images = set(Path(os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/images/val')).glob('*.jpg'))
existing_images = {path.name for path in existing_train_images.union(existing_val_images)}

# 过滤掉已存在的图像文件
filtered_image_files = [image_path for image_path in image_files if image_path.name not in existing_images]

if not filtered_image_files:
    print("所有图像文件均已存在于目标文件夹中，无需进一步处理。")
    exit()

# 随机打乱文件列表
random.shuffle(filtered_image_files)

# 计算训练集和验证集的数量
train_count = int(len(filtered_image_files) * train_ratio)
train_images = filtered_image_files[:train_count]
val_images = filtered_image_files[train_count:]


# 移动或复制文件到相应位置
def copy_files(files, dst_image_dir, dst_label_dir):
    for image_path in files:
        label_path = Path(original_labels_dir) / f"{image_path.stem}.txt"
        dst_image_path = Path(dst_image_dir) / image_path.name

        if label_path.exists():
            shutil.copy(image_path, dst_image_dir)
            shutil.copy(label_path, dst_label_dir)
            print(f"已复制 {image_path.name} 和对应的标签文件到 {dst_image_dir}")
        else:
            print(f"警告: 没有找到 {label_path} 对应的标签文件")


# 创建必要的目录
os.makedirs(os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/images/train'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/labels/train'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/images/val'), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/labels/val'), exist_ok=True)

# 执行复制操作
copy_files(train_images, os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/images/train'),
           os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/labels/train'))
copy_files(val_images, os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/images/val'),
           os.path.join(os.path.dirname(__file__), f'{DATASET_OUTPUT_FOLDER}/labels/val'))

print("图像和标注文件已移动/复制完成")
