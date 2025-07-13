import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 显示负号

# 生成带噪声的线性数据
np.random.seed(42)
x = np.linspace(0, 10, 20)
y_true = 2.5 * x + 1.2  # 真实函数
y = y_true + np.random.normal(scale=3, size=len(x))  # 添加噪声

# 最小二乘法计算函数
def least_squares(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)
    
    # 计算斜率和截距
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b = (sum_y - m * sum_x) / n
    
    # 计算预测值和残差
    y_pred = m * x + b
    residuals = y - y_pred
    
    return m, b, y_pred, residuals

# 计算拟合结果
m, b, y_pred, residuals = least_squares(x, y)

# 创建画布和子图布局
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2)

# 主图：数据点和拟合线
ax_main = fig.add_subplot(gs[0, 0])
ax_main.scatter(x, y, c='blue', label='观测数据', zorder=5)
ax_main.plot(x, y_true, 'g--', linewidth=2, alpha=0.7, label='真实关系')
line, = ax_main.plot([], [], 'r-', linewidth=2, label='拟合直线')
ax_main.set_xlim(min(x)-1, max(x)+1)
ax_main.set_ylim(min(y)-5, max(y)+5)
ax_main.set_xlabel('X')
ax_main.set_ylabel('Y')
ax_main.set_title('最小二乘法线性拟合')
ax_main.grid(True, linestyle='--', alpha=0.7)
ax_main.legend(loc='upper left')

# 残差图
ax_res = fig.add_subplot(gs[1, 0])
res_plot = ax_res.scatter([], [], c='red', s=40, label='残差')
ax_res.axhline(0, color='gray', linestyle='--')
ax_res.set_xlim(min(x)-1, max(x)+1)
ax_res.set_ylim(min(residuals)-1, max(residuals)+1)
ax_res.set_xlabel('X')
ax_res.set_ylabel('残差')
ax_res.set_title('残差分布图')
ax_res.grid(True, linestyle='--', alpha=0.7)
ax_res.legend()

# 3D损失函数曲面图
ax_3d = fig.add_subplot(gs[:, 1], projection='3d')
ax_3d.set_xlabel('斜率 m')
ax_3d.set_ylabel('截距 b')
ax_3d.set_zlabel('残差平方和')
ax_3d.set_title('损失函数曲面')

# 生成网格数据用于3D曲面
m_range = np.linspace(m-5, m+5, 50)
b_range = np.linspace(b-10, b+10, 50)
M, B = np.meshgrid(m_range, b_range)
SSE = np.zeros_like(M)

# 计算每个(m,b)组合的残差平方和
for i in range(len(m_range)):
    for j in range(len(b_range)):
        y_hat = M[j, i] * x + B[j, i]
        SSE[j, i] = np.sum((y - y_hat)**2)

# 绘制3D曲面
ax_3d.plot_surface(M, B, SSE, cmap='viridis', alpha=0.7)
ax_3d.contour(M, B, SSE, zdir='z', offset=np.min(SSE), cmap='coolwarm')

# 标记最小点
min_point = ax_3d.scatter([m], [b], [np.sum(residuals**2)], 
                         c='red', s=100, label='最小点')

# 添加箭头和标注
arrow_props = dict(arrowstyle="->", color='black', lw=1.5)
ax_main.annotate('残差 = 观测值 - 预测值', 
                xy=(x[5], y[5]), 
                xytext=(x[5]-2, y[5]+5),
                arrowprops=arrow_props)

# 动画更新函数
def update(frame):
    # 动态绘制拟合线
    if frame < len(x):
        line.set_data(x[:frame+1], y_pred[:frame+1])
    
    # 动态绘制残差
    if frame >= len(x) and frame < 2*len(x):
        idx = frame - len(x)
        res_points = np.array([[x[idx], y[idx]], [x[idx], y_pred[idx]]])
        res_plot.set_offsets(np.c_[x[:idx+1], residuals[:idx+1]])
        if idx == 0:
            ax_res.add_patch(plt.Line2D([x[0], x[0]], [0, residuals[0]], 
                                      color='purple', alpha=0.5))
        else:
            ax_res.add_patch(plt.Line2D([x[idx], x[idx]], [0, residuals[idx]], 
                                      color='purple', alpha=0.5))
    
    # 更新3D点
    if frame == 2*len(x):
        min_point.set_sizes([100])
        ax_3d.text(m, b, np.sum(residuals**2), 
                  f'最小点: m={m:.2f}, b={b:.2f}\nSSE={np.sum(residuals**2):.2f}', 
                  color='red')
    
    return line, res_plot, min_point

# 创建动画
ani = FuncAnimation(fig, update, frames=2*len(x)+1, interval=500, blit=False)

# 添加公式说明
plt.figtext(0.1, 0.02, 
            r'最小二乘法公式: $m = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}$  $b = \frac{\sum y - m\sum x}{n}$',
            fontsize=12)
plt.figtext(0.1, 0.00, 
            f'计算结果: 斜率 m = {m:.4f}, 截距 b = {b:.4f}, 残差平方和 = {np.sum(residuals**2):.2f}',
            fontsize=12, color='red')

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.show()