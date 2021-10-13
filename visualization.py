"""
可视化提供的Nemo鱼不同斑点的像素分布情况
"""
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 加载样本数据
array_sample = scio.loadmat('array_sample.mat')
array_sample = array_sample['array_sample']

# 绘制Nemo鱼颜色斑点统计直方图(灰度图)
orange_color_idx = array_sample[:, 4] == 1
white_color_idx = array_sample[:, 4] == -1

# 获取灰度值数据0-1之间
orange_color_data = array_sample[orange_color_idx, 0]
white_color_data = array_sample[white_color_idx, 0]

# 估计均值和方差
orange_mu = np.mean(orange_color_data)
# orange_sigma = np.sqrt(np.mean(np.power(orange_color_data - orange_mu, 2)))
orange_sigma = np.std(orange_color_data)

white_mu = np.mean(white_color_data)
# white_sigma = np.sqrt(np.mean(np.power(white_color_data - white_mu, 2)))
white_sigma = np.std(white_color_data)

# 设置格子数
num_bins = 30

# 从直方图看出两种颜色的分布不太像正态分布，后面可能需要考虑其他方法进行参数估计
plt.figure('Gray_PDF')
_, orange_bins, _ = plt.hist(orange_color_data, num_bins, density=True, color='orange', alpha=0.6, label='orange_pixel')
_, white_bins, _ = plt.hist(white_color_data, num_bins, density=True, color='blue', alpha=0.6, label='white_pixel')
orange_y = norm.pdf(orange_bins, orange_mu, orange_sigma)  
white_y = norm.pdf(white_bins, white_mu, white_sigma)

plt.plot(orange_bins, orange_y, 'r--', label='orange_pdf')  
plt.plot(white_bins, white_y, 'c--', label='white_pdf')  

plt.xlabel('Norm pixel')
plt.ylabel('Probability density')
plt.title('Pixel distribution histogram')
plt.legend()
plt.show()  # 拟合曲线有较大的部分重叠，有一些像素过多偏离阈值可考虑剔除部分拟合样本，再进行最大似然估计参数


# 获取RGB数据，数据值位于0-1之间
orange_color_data = array_sample[orange_color_idx, 1:4]
white_color_data = array_sample[white_color_idx, 1:4]


# 绘制Nemo鱼颜色斑点统计直方图(彩色图)
fig = plt.figure('RGB_PDF')
ax = fig.add_subplot(111, projection='3d')
i = 0
for c, z in zip(['r', 'g', 'b'], [30, 20, 10]):

	bins = 30
	xs_o = np.histogram(orange_color_data[:, i], density=True, bins=bins)[1]
	ys_o = np.histogram(orange_color_data[:, i], density=True, bins=bins)[0]

	xs_w = np.histogram(white_color_data[:, i], density=True, bins=bins)[1]
	ys_w = np.histogram(white_color_data[:, i], density=True, bins=bins)[0]	

	xs_o = xs_o[:30]
	xs_w = xs_w[:30]
	# You can provide either a single color or an array. To demonstrate this,
	# the first bar of each set will be colored cyan.

	ax.bar(xs_o, ys_o, zs=z, zdir='y', width=0.03, color=c, alpha=1.0)
	ys_pdf = norm.pdf(xs_o, np.mean(orange_color_data[:, i]), np.std(orange_color_data[:, i]))
	ax.plot(xs_o, ys_pdf, zs=z, zdir='y')

	ax.bar(xs_w, ys_w, zs=z, zdir='y', width=0.03, color=c, alpha=0.4)
	ys_pdf = norm.pdf(xs_w, np.mean(white_color_data[:, i]), np.std(white_color_data[:, i]))
	ax.plot(xs_w, ys_pdf, zs=z, zdir='y')
	i += 1

ax.set_xlabel('Norm pixel')
ax.set_ylabel('Space')
ax.set_zlabel('Probability density')
ax.set_title('RGB pixel distribution histogram')
plt.show()