import cv2
import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch.nn.functional as F
from scipy.stats import entropy

def infer_image_with_features(model, raw_image, input_size=518):
    """
    处理输入图像，提取深度图，并返回 DINOv2 编码器的不同层特征。
    """
    # 预处理图像（Resize, Normalize）
    image, (h, w) = model.image2tensor(raw_image, input_size)

    # 提取 ViT 中间层特征
    features = model.pretrained.get_intermediate_layers(
        image, 
        n=[2,5,8,11],  # 获取第 4、11、17、23 层特征
        reshape=True,  # 转换为 (B, C, H, W) 格式
        return_class_token=False
    )

    # 计算最终深度图
    depth = model.forward(image)
    depth = torch.nn.functional.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)[0, 0]

    return depth.detach().cpu().numpy(), [feat.detach().cpu().numpy() for feat in features]

def visualize_features(feature_maps, mask, layers, depth_map,raw_img,file_name):
    """
    显示不同层的 ViT 特征图，并在 entropy 图上绘制 mask 轮廓。
    """
    # fig, axes = plt.subplots(1, len(feature_maps) + 2, figsize=(25, 6))  # 设置图像大小
    # plt.subplots_adjust(wspace=0.4)  # 调整子图间距

    # 调整 mask 以匹配特征图的大小
    resized_mask = resize_mask(mask, (feature_maps[0].shape[3], feature_maps[0].shape[2]))

    # 提取轮廓
    mask_contours, _ = cv2.findContours(resized_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在深度图上绘制原始 mask 轮廓
    cv2.drawContours(depth_map, original_contours, -1, (0, 255, 0), 6)

    # 显示深度图
    # ax = axes[0]
    # ax.imshow(raw_img, cmap='inferno')
    # ax.set_title("Raw Fig", fontsize=16)  # 设置字体大小
    # ax.axis("off")

    data = [file_name]

    for i, feature_map in enumerate(feature_maps):
        feature_map = torch.tensor(feature_map).squeeze()   # 从 ndarray 转换为 Tensor
  
        # 计算 Shannon 熵
        glass_entropy, all_entropy,other_entropy = calculate_shannon_entropy(feature_map, resized_mask)

        # **转换 entropy 图为 NumPy 并归一化**
        all_entropy_np = all_entropy.numpy()
        all_entropy_norm = cv2.normalize(all_entropy_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        all_entropy_color = cv2.applyColorMap(all_entropy_norm, cv2.COLORMAP_HOT)  # 伪彩色映射

        # **在 entropy 图上绘制 mask 轮廓**
        cv2.drawContours(all_entropy_color, mask_contours, -1, (255, 0, 0), 1)

        # **显示 entropy 图**
        # ax = axes[i + 1]
        # im = ax.imshow(all_entropy_color)
        glass_value = glass_entropy.sum()/glass_entropy.count_nonzero()
        other_value = other_entropy.sum()/other_entropy.count_nonzero()
        data.append(round(glass_value.item(),3))
        data.append(round(other_value.item(),3))
        # ax.set_title(f"Layer {layers[i]}\nGlass entropy:{glass_value:.3f}, Other:{other_value:.3f}", 
        #              fontsize=14)  # 设置标题字体大小
        # ax.axis("off")

    # **添加 colorbar**
    # ax = axes[5]
    # cmap = plt.colormaps.get_cmap('Spectral_r')
    # ax.imshow(depth_map, cmap=cmap)
    # ax.set_title("Depth Pred Fig", fontsize=16)  # 设置字体大小
    # ax.axis("off")

    # plt.savefig(f"depth_vis/feature_entropies_{file_name}.png", dpi=150, bbox_inches='tight')  # 高分辨率保存
    # plt.close(fig)
    return data



def resize_mask(mask,feature_shape):
   mask_resized = cv2.resize(mask,feature_shape,interpolation=cv2.INTER_NEAREST)
   return mask_resized

def calculate_shannon_entropy(feature_map,resized_mask):
    mask = (resized_mask>0).astype(np.uint8)   #transform to 0/1 mask
    mask = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).expand(feature_map.shape)    #(1,h,w)

    c, h, w = feature_map.shape  

    entropy_glass, entropy = torch.zeros(h, w), torch.zeros(h, w)

    for x in range(h):
        for y in range(w):
            masked_values = feature_map[:, x, y][mask[:, x, y]]
            other_values = feature_map[:, x, y]
            mean_map = np.mean(feature_map,axis=0)
            soft_mask,soft_other = F.softmax(masked_values,dim=0),F.softmax(other_values,dim=0)
            entropy_mask,entropy_other = -torch.sum(soft_mask * torch.log(soft_mask + 1e-10)),-torch.sum(soft_other * torch.log(soft_other + 1e-10))
            entropy_glass[x,y],entropy[x,y] = entropy_mask,entropy_other
    return entropy_glass,entropy,entropy-entropy_glass,mean_map      #glass,all,all but glass

def calculate_l2_distance(feature_maps,feature_maps_sharpened):
    size = len(feature_maps)
    H,W = feature_maps[0].shape[3],feature_maps[0].shape[2]
    dist = torch.zeros(size,H,W)
    for i in range(size):
        dist[i,:,:] = torch.norm(feature_maps[i] - feature_maps_sharpened[i],p=2,dim=1).squeeze(0)
    return dist






   
