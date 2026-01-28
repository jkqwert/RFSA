# -*- coding: utf-8 -*-
'''
Plot relationship validity verification comparative experiments
绘制关系有效性验证对比实验图表
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# 设置中文字体和样式
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体显示中文
rcParams['axes.unicode_minus'] = False  # 正常显示负号
rcParams['font.size'] = 11

# 数据：文本到图像 (Text-to-Image)
t2i_data = {
    'Original': [46.85, 75.68, 84.68],
    'Text Swap': [8.11, 21.62, 25.23],
    'Image Swap': [23.42, 45.05, 54.05],
    'Both Swap': [20.72, 36.04, 51.35]
}

# 数据：图像到文本 (Image-to-Text)
i2t_data = {
    'Original': [54.92, 89.75, 95.08],
    'Text Swap': [9.02, 24.18, 29.51],
    'Image Swap': [31.56, 55.74, 61.89],
    'Both Swap': [28.28, 49.59, 55.33]
}

# Recall@K 标签
recall_metrics = ['Recall@1', 'Recall@5', 'Recall@10']
x_positions = np.arange(len(recall_metrics))

# 颜色方案（使用专业的配色）
colors = {
    'Original': '#2E7D32',      # 深绿色
    'Text Swap': '#D32F2F',     # 深红色
    'Image Swap': '#1976D2',    # 深蓝色
    'Both Swap': '#F57C00'      # 深橙色
}

# 标记样式
markers = {
    'Original': 'o',
    'Text Swap': 's',
    'Image Swap': '^',
    'Both Swap': 'D'
}

# 创建图形，1行2列
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 文本到图像 (Text-to-Image)
for label, values in t2i_data.items():
    ax1.plot(x_positions, values, 
             marker=markers[label], 
             color=colors[label],
             linewidth=2.5, 
             markersize=8,
             label=label,
             linestyle='-',
             markeredgewidth=1.5,
             markeredgecolor='white')

ax1.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax1.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax1.set_title('Text-to-Image Retrieval', fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x_positions)
ax1.set_xticklabels(recall_metrics)
ax1.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax1.set_ylim(0, 105)

# 添加数值标签 - 所有数据点
for label, values in t2i_data.items():
    for i, v in enumerate(values):
        # 根据值的大小调整标签位置，避免重叠
        offset = 3 if v > 50 else 2.5
        ax1.text(x_positions[i], v + offset, f'{v:.1f}', 
                ha='center', va='bottom', fontsize=7.5, 
                color=colors[label], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=colors[label], alpha=0.7, linewidth=0.8))

# 图2: 图像到文本 (Image-to-Text)
for label, values in i2t_data.items():
    ax2.plot(x_positions, values, 
             marker=markers[label], 
             color=colors[label],
             linewidth=2.5, 
             markersize=8,
             label=label,
             linestyle='-',
             markeredgewidth=1.5,
             markeredgecolor='white')

ax2.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax2.set_ylabel('Recall (%)', fontsize=12, fontweight='bold')
ax2.set_title('Image-to-Text Retrieval', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x_positions)
ax2.set_xticklabels(recall_metrics)
ax2.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
ax2.set_ylim(0, 105)

# 添加数值标签 - 所有数据点
for label, values in i2t_data.items():
    for i, v in enumerate(values):
        # 根据值的大小调整标签位置，避免重叠
        offset = 3 if v > 50 else 2.5
        ax2.text(x_positions[i], v + offset, f'{v:.1f}', 
                ha='center', va='bottom', fontsize=7.5, 
                color=colors[label], fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=colors[label], alpha=0.7, linewidth=0.8))

# 调整布局
plt.tight_layout()

# 保存图形
output_path = 'd:/python project/sorclip/outputs/relationship_validity_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'图表已保存到: {output_path}')

# 同时保存为PDF格式（用于论文）
pdf_path = 'd:/python project/sorclip/outputs/relationship_validity_comparison.pdf'
plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'PDF版本已保存到: {pdf_path}')

# 显示图形
plt.show()

print('\n图表绘制完成！')
print('\n数据摘要:')
print('=' * 60)
print('文本到图像 (Text-to-Image):')
for label, values in t2i_data.items():
    print(f'  {label:15s}: Recall@1={values[0]:5.2f}%, Recall@5={values[1]:5.2f}%, Recall@10={values[2]:5.2f}%')

print('\n图像到文本 (Image-to-Text):')
for label, values in i2t_data.items():
    print(f'  {label:15s}: Recall@1={values[0]:5.2f}%, Recall@5={values[1]:5.2f}%, Recall@10={values[2]:5.2f}%')
print('=' * 60)
