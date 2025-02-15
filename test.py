import os
import cv2
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm  # 进度条
from PIL import Image
plt.switch_backend('Agg')
from depth_anything_v2.dpt import DepthAnythingV2
from feature_tools import *

# 选择设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# 配置不同 ViT 版本的参数
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

column_names = ["Img name","Layer2 In","Layer2 Out","Layer5 In","Layer5 Out","Layer8 In","Layer8 Out","Layer11 In","Layer11 Out"]
rows = []

encoder = 'vitb'  # 选择 ViT 版本: 'vits', 'vitb', 'vitl', 'vitg'

# 加载模型
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cuda'))
model = model.to(DEVICE).eval()

# 文件夹路径
photo_dir = "photos"
mask_dir = "mask"
output_dir = "depth_vis"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取所有图片文件
photo_files = sorted([f for f in os.listdir(photo_dir) if f.endswith(('.jpg', '.png'))])

count = 0

# 进度条，显示进度
for photo_file in tqdm(photo_files, desc="Processing Images", unit="image"):
    # 获取文件名（不带扩展名）
    file_name = os.path.splitext(photo_file)[0]
    
    # 加载图像和对应的 mask
    photo_path = os.path.join(photo_dir, photo_file)
    mask_path = os.path.join(mask_dir, f"{file_name}.png")  # 假设 mask 是 PNG 格式
    
    if not os.path.exists(mask_path):
        print(f"⚠️ 跳过 {photo_file}，未找到对应的 mask 文件")
        continue
    
    raw_img = cv2.imread(photo_path)
    image = Image.open(photo_path).convert('RGB')
    img_rgb = np.array(image)
    center_value = -4
    sharpen_kernel = np.array([
        [0, -center_value / 4, 0],
        [-center_value / 4, center_value, -center_value / 4],
        [0, -center_value / 4, 0]
    ], dtype=np.float32)
    sharpened_img = cv2.filter2D(img_rgb, -1, sharpen_kernel)
    sharpened_img_rgb = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 获取深度图和特征
    depth_map, feature_maps = infer_image_with_features(model, raw_img)
    _,sharpened_feature_maps = infer_image_with_features(model,sharpened_img_rgb)

    dist_map = calculate_l2_distance(torch.tensor(np.array(feature_maps)),torch.tensor(np.array(sharpened_feature_maps)))

    # 显示不同层的特征图
    layers = [2, 5, 8, 11]
    line = visualize_features(feature_maps, mask, layers, depth_map, raw_img, file_name,dist_map)
    rows.append(line)

    count += 1
    if count > 5:
        break


print("✅ 批量处理完成！")
print("Start writing into csv...")
with open("output.csv",mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow(column_names)
    for row in rows:
        writer.writerow(row)
print("Done!")
