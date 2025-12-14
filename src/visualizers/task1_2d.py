import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from config import cfg

def run(data, models):
    print("🎨 [Task 1] 生成 2D 9分类器对比图...")
    X, y = data['X_full'], data['y_full'] # 使用全部数据绘图
    
    # 创建网格
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, cfg.RES_2D),
        np.linspace(y_min, y_max, cfg.RES_2D)
    )

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    cmap_bg = ListedColormap(cfg.BG_COLORS) # 确保三色清晰

    for i, (name, model) in enumerate(models.items()):
        ax = axes[i]
        model.fit(X, y) # 在全数据上fit以展示边界
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.6)
        
        # 绘制散点
        for cls_idx, cls_name in enumerate(cfg.CLASS_NAMES):
            idx = np.where(y == cls_idx)
            ax.scatter(X[idx, 0], X[idx, 1], c=cfg.COLORS[cls_idx], 
                       label=cls_name, edgecolor='k', s=30)
        
        ax.set_title(name)
        if i == 0: ax.legend(loc='upper left', fontsize='x-small')
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/task1_2d_comparison.png")
    print(f"✅ 保存完毕: {cfg.OUTPUT_DIR}/task1_2d_comparison.png")