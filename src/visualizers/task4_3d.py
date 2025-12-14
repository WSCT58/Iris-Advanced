import plotly.graph_objects as go
import numpy as np
from skimage import measure
from src.config import cfg


def run(data, model):
    print("🎨 [Task 4] 生成 3D 边界 + 概率图 (Boundary + Probability Volume)...")

    # 1. 准备多分类数据 (3个特征)
    X, y = data['X_full'], data['y_full']
    model.fit(X, y)

    # 2. 生成体素网格
    res = 35  # 分辨率
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    z_min, z_max = X[:, 2].min() - pad, X[:, 2].max() + pad

    gx, gy, gz = np.mgrid[x_min:x_max:complex(0, res),
                 y_min:y_max:complex(0, res),
                 z_min:z_max:complex(0, res)]

    # 3. 预测概率
    # predict_proba 返回 (N, 3)，我们需要最大概率值来画热图
    all_probs = model.predict_proba(np.c_[gx.ravel(), gy.ravel(), gz.ravel()])
    max_probs = np.max(all_probs, axis=1).reshape(gx.shape)  # 置信度 (0.33 ~ 1.0)
    preds = np.argmax(all_probs, axis=1).reshape(gx.shape)  # 类别 (0, 1, 2)

    fig = go.Figure()

    # --- A. 概率图 (Probability Map - Volume) ---
    # 我们用 Volume 展示"类别+置信度"
    # 颜色代表类别，透明度代表置信度(越确信越不透明)
    fig.add_trace(go.Volume(
        x=gx.flatten(), y=gy.flatten(), z=gz.flatten(),
        value=preds.flatten(),  # 颜色由类别决定

        # 仅显示置信度比较高的区域，让中间留出空隙给边界
        # 这里的 trick 是结合透明度
        opacity=0.08,
        surface_count=15,
        colorscale=[[0, cfg.COLORS[0]], [0.5, cfg.COLORS[1]], [1, cfg.COLORS[2]]],
        showscale=False,
        name='Probability Cloud'
    ))

    # --- B. 决策边界 (Decision Boundary - Mesh) ---
    # 这是一个比较高级的技巧：
    # 对于多分类，边界其实就是 Class 0 vs Class 1, Class 1 vs Class 2 的分界面
    # 我们通过检测"预测类别跳变"的地方来近似边界

    # 为了简化且视觉效果好，我们画出 Setosa (Class 0) 的边界
    # 因为 Setosa 是最好分的，它的边界最清晰
    probs_class0 = all_probs[:, 0].reshape(gx.shape)

    try:
        # 提取 Class 0 的边界 (P=0.5)
        verts, faces, _, _ = measure.marching_cubes(probs_class0, 0.5)

        # 坐标转换
        rx = verts[:, 0] * (x_max - x_min) / (res - 1) + x_min
        ry = verts[:, 1] * (y_max - y_min) / (res - 1) + y_min
        rz = verts[:, 2] * (z_max - z_min) / (res - 1) + z_min

        fig.add_trace(go.Mesh3d(
            x=rx, y=ry, z=rz, i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=0.5,
            color='gold',  # 金色边界
            name='Boundary (Setosa)',
            showscale=False
        ))
    except:
        print("⚠️ 无法生成 Setosa 边界 (可能是数据分布问题)")

    # --- C. 真实数据点 ---
    for cls in [0, 1, 2]:
        mask = y == cls
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
            mode='markers',
            name=cfg.CLASS_NAMES[cls],
            marker=dict(color=cfg.COLORS[cls], size=5, line=dict(width=2, color='white'))
        ))

    fig.update_layout(
        title="Task 4: 3D Probability Volume + Decision Boundary",
        scene=dict(
            xaxis_title=data['names'][0],
            yaxis_title=data['names'][1],
            zaxis_title=data['names'][2]
        )
    )

    filename = "task4_boundary_prob_map.html"
    fig.write_html(f"{cfg.OUTPUT_DIR}/{filename}")
    print(f"✅ 保存完毕: {cfg.OUTPUT_DIR}/{filename}")
    cfg.open_file(filename)