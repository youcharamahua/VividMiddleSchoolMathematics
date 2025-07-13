import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

# ����ȫ������
plt.rcParams['font.sans-serif'] = ['SimHei']  # ����������ʾ����
plt.rcParams['axes.unicode_minus'] = False    # ��ʾ����

# ���ɴ���������������
np.random.seed(42)
x = np.linspace(0, 10, 20)
y_true = 2.5 * x + 1.2  # ��ʵ����
y = y_true + np.random.normal(scale=3, size=len(x))  # �������

# ��С���˷����㺯��
def least_squares(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x**2)
    
    # ����б�ʺͽؾ�
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b = (sum_y - m * sum_x) / n
    
    # ����Ԥ��ֵ�Ͳв�
    y_pred = m * x + b
    residuals = y - y_pred
    
    return m, b, y_pred, residuals

# ������Ͻ��
m, b, y_pred, residuals = least_squares(x, y)

# ������������ͼ����
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2)

# ��ͼ�����ݵ�������
ax_main = fig.add_subplot(gs[0, 0])
ax_main.scatter(x, y, c='blue', label='�۲�����', zorder=5)
ax_main.plot(x, y_true, 'g--', linewidth=2, alpha=0.7, label='��ʵ��ϵ')
line, = ax_main.plot([], [], 'r-', linewidth=2, label='���ֱ��')
ax_main.set_xlim(min(x)-1, max(x)+1)
ax_main.set_ylim(min(y)-5, max(y)+5)
ax_main.set_xlabel('X')
ax_main.set_ylabel('Y')
ax_main.set_title('��С���˷��������')
ax_main.grid(True, linestyle='--', alpha=0.7)
ax_main.legend(loc='upper left')

# �в�ͼ
ax_res = fig.add_subplot(gs[1, 0])
res_plot = ax_res.scatter([], [], c='red', s=40, label='�в�')
ax_res.axhline(0, color='gray', linestyle='--')
ax_res.set_xlim(min(x)-1, max(x)+1)
ax_res.set_ylim(min(residuals)-1, max(residuals)+1)
ax_res.set_xlabel('X')
ax_res.set_ylabel('�в�')
ax_res.set_title('�в�ֲ�ͼ')
ax_res.grid(True, linestyle='--', alpha=0.7)
ax_res.legend()

# 3D��ʧ��������ͼ
ax_3d = fig.add_subplot(gs[:, 1], projection='3d')
ax_3d.set_xlabel('б�� m')
ax_3d.set_ylabel('�ؾ� b')
ax_3d.set_zlabel('�в�ƽ����')
ax_3d.set_title('��ʧ��������')

# ����������������3D����
m_range = np.linspace(m-5, m+5, 50)
b_range = np.linspace(b-10, b+10, 50)
M, B = np.meshgrid(m_range, b_range)
SSE = np.zeros_like(M)

# ����ÿ��(m,b)��ϵĲв�ƽ����
for i in range(len(m_range)):
    for j in range(len(b_range)):
        y_hat = M[j, i] * x + B[j, i]
        SSE[j, i] = np.sum((y - y_hat)**2)

# ����3D����
ax_3d.plot_surface(M, B, SSE, cmap='viridis', alpha=0.7)
ax_3d.contour(M, B, SSE, zdir='z', offset=np.min(SSE), cmap='coolwarm')

# �����С��
min_point = ax_3d.scatter([m], [b], [np.sum(residuals**2)], 
                         c='red', s=100, label='��С��')

# ��Ӽ�ͷ�ͱ�ע
arrow_props = dict(arrowstyle="->", color='black', lw=1.5)
ax_main.annotate('�в� = �۲�ֵ - Ԥ��ֵ', 
                xy=(x[5], y[5]), 
                xytext=(x[5]-2, y[5]+5),
                arrowprops=arrow_props)

# �������º���
def update(frame):
    # ��̬���������
    if frame < len(x):
        line.set_data(x[:frame+1], y_pred[:frame+1])
    
    # ��̬���Ʋв�
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
    
    # ����3D��
    if frame == 2*len(x):
        min_point.set_sizes([100])
        ax_3d.text(m, b, np.sum(residuals**2), 
                  f'��С��: m={m:.2f}, b={b:.2f}\nSSE={np.sum(residuals**2):.2f}', 
                  color='red')
    
    return line, res_plot, min_point

# ��������
ani = FuncAnimation(fig, update, frames=2*len(x)+1, interval=500, blit=False)

# ��ӹ�ʽ˵��
plt.figtext(0.1, 0.02, 
            r'��С���˷���ʽ: $m = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}$  $b = \frac{\sum y - m\sum x}{n}$',
            fontsize=12)
plt.figtext(0.1, 0.00, 
            f'������: б�� m = {m:.4f}, �ؾ� b = {b:.4f}, �в�ƽ���� = {np.sum(residuals**2):.2f}',
            fontsize=12, color='red')

plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.show()