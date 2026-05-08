from skimage import data, transform
import matplotlib.pyplot as plt
import numpy as np

# 加载测试图像
image = data.coffee()
print("原始图像形状:", image.shape)
print("图像类型:", type(image))

# 下采样因子
ratio = 20

# 使用局部均值下采样
image1 = transform.downscale_local_mean(image, (ratio, ratio, 1))

# 转换回 uint8 以便正确显示
image1 = np.clip(image1, 0, 255).astype(np.uint8)

print("下采样后形状:", image1.shape)

# 使用 matplotlib 显示图像
plt.imshow(image1)
plt.show()