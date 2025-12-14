import plotly.graph_objects as go
import numpy as np
from skimage import measure
from src.config import cfg

def run(data, model):
    print("🎨 [Task 2] 生成 3D 交互式决策边界 (HTML)...")
    X, y = data['X_full'], data['y_full']
    model.fit(X, y)

    # 3D 网格
    pad = 0.5
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    z_min, z_max = X[:, 2].min()-pad, X[:, 2].max()+pad

    gx, gy, gz = np.mgrid[x_min:x_max:complex(0, cfg.RES_3D),
                          y_min:y_max:complex(0, cfg.RES_3D),
                          z_min:z_max:complex(0, cfg.RES_3D)]

    probs = model.predict_proba(np.c_[gx.ravel(), gy.ravel(), gz.ravel()])[:, 1].reshape(gx.shape)

    # Marching Cubes 提取曲面
    verts, faces, _, _ = measure.marching_cubes(probs, 0.5)

    # 坐标反变换 (Grid Index -> Real World Coords)
    real_x = verts[:, 0] * (x_max - x_min) / (cfg.RES_3D - 1) + x_min
    real_y = verts[:, 1] * (y_max - y_min) / (cfg.RES_3D - 1) + y_min
    real_z = verts[:, 2] * (z_max - z_min) / (cfg.RES_3D - 1) + z_min

    fig = go.Figure()
    # 边界曲面
    fig.add_trace(go.Mesh3d(x=real_x, y=real_y, z=real_z,
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            opacity=0.6, color='gold', name='Boundary'))
    # 数据点
    for cls in [0, 1]:
        mask = y == cls
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0], y=X[mask, 1], z=X[mask, 2], mode='markers',
            name=cfg.CLASS_NAMES[cls], marker=dict(color=cfg.COLORS[cls], size=5)
        ))

    fig.write_html(f"{cfg.OUTPUT_DIR}/task2_interactive.html")
    print(f"✅ 保存完毕: {cfg.OUTPUT_DIR}/task2_interactive.html")
    cfg.open_file("task2_interactive.html")