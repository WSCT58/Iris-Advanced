import plotly.graph_objects as go
import numpy as np
from config import cfg

def run(data, model):
    print("🎨 [Task 4] 生成全息体渲染投影 (Holographic Volume)...")
    X, y = data['X_full'], data['y_full']
    model.fit(X, y) # 多分类

    # 生成高密度网格用于体渲染
    res = cfg.RES_VOL
    gx, gy, gz = np.mgrid[X[:,0].min():X[:,0].max():complex(0, res), 
                          X[:,1].min():X[:,1].max():complex(0, res), 
                          X[:,2].min():X[:,2].max():complex(0, res)]
    
    # 预测整个空间
    probs = model.predict_proba(np.c_[gx.ravel(), gy.ravel(), gz.ravel()])
    preds = np.argmax(probs, axis=1).reshape(gx.shape)

    fig = go.Figure()

    # 1. 体渲染 (彩色果冻云)
    fig.add_trace(go.Volume(
        x=gx.flatten(), y=gy.flatten(), z=gz.flatten(),
        value=preds.flatten(),
        isomin=0, isomax=2, opacity=0.08, surface_count=15,
        colorscale=[[0, cfg.COLORS[0]], [0.5, cfg.COLORS[1]], [1, cfg.COLORS[2]]],
        showscale=False, name='Decision Cloud'
    ))

    # 2. 真实数据点
    for cls in [0, 1, 2]:
        mask = y == cls
        fig.add_trace(go.Scatter3d(
            x=X[mask,0], y=X[mask,1], z=X[mask,2], mode='markers',
            name=cfg.CLASS_NAMES[cls], marker=dict(color=cfg.COLORS[cls], size=6, line=dict(width=2, color='white'))
        ))

    # 3. 墙面投影 (增加空间感)
    # 底部投影
    fig.add_trace(go.Scatter3d(
        x=X[:,0], y=X[:,1], z=np.full_like(X[:,2], X[:,2].min()),
        mode='markers', marker=dict(color=y, colorscale=[cfg.COLORS[0], cfg.COLORS[1], cfg.COLORS[2]], opacity=0.2),
        showlegend=False
    ))

    fig.update_layout(title="Holographic Decision Cube", scene=dict(aspectmode='data'))
    fig.write_html(f"{cfg.OUTPUT_DIR}/task4_holographic.html")
    print(f"✅ 保存完毕: {cfg.OUTPUT_DIR}/task4_holographic.html")