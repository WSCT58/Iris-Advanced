import plotly.graph_objects as go
import numpy as np
from src.config import cfg


def run(data, model):
    print("🎨 [Task 3] 生成 3D 概率曲面图 (Probability Surface)...")

    # --- 1. 数据准备 ---
    # 注意：为了让 Z轴表示概率，我们必须只能用 2个特征 (X, Y)
    # 我们使用 config 中定义的 FEATURES_2D (通常是花瓣长、花瓣宽，区分度最好)
    feat_indices = cfg.FEATURES_2D
    X = data['X_full'][:, [0, 1]]  # 这里实际上取的是 data_manager 处理后的对应列，通常就是 Petal Length/Width
    y = data['y_full']

    # 只要二分类数据 (Class 0 vs Class 1)
    mask = y < 2
    X = X[mask]
    y = y[mask]

    # 重新训练模型 (只用这2个特征)
    model.fit(X, y)

    # --- 2. 生成网格 ---
    res = 50
    pad = 0.5
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad

    # 生成平面网格
    gx = np.linspace(x_min, x_max, res)
    gy = np.linspace(y_min, y_max, res)
    xx, yy = np.meshgrid(gx, gy)

    # --- 3. 计算 Z轴 (概率) ---
    # 预测网格中每个点的概率
    zz = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    zz = zz.reshape(xx.shape)  # 形状变成 (50, 50)

    fig = go.Figure()

    # --- 核心绘制 1: S型概率曲面 ---
    fig.add_trace(go.Surface(
        x=gx, y=gy, z=zz,
        colorscale='RdBu',  # 红-白-蓝
        opacity=0.8,  # 半透明，以便看到后面的点
        name='Probability Surface',
        showscale=True,
        colorbar=dict(title="Probability P(Class=1)"),
        # 在地板和墙壁上投射等高线 (关键效果！)
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project=dict(z=True))
        )
    ))

    # --- 核心绘制 2: 真实数据点 ---
    # 这一步很关键：
    # Class 0 的点，真实概率是 0，所以画在 Z=0 的位置 (地板)
    # Class 1 的点，真实概率是 1，所以画在 Z=1 的位置 (天花板)
    # 这样可以看出曲面拟合得好不好

    # 画 Class 0 (Setosa) -> 红色，在地板
    mask0 = (y == 0)
    fig.add_trace(go.Scatter3d(
        x=X[mask0, 0], y=X[mask0, 1], z=np.zeros(sum(mask0)),  # Z=0
        mode='markers',
        name=f"{cfg.CLASS_NAMES[0]} (True=0)",
        marker=dict(color='red', size=6, line=dict(width=2, color='white'))
    ))

    # 画 Class 1 (Versicolor) -> 蓝色，在天花板
    mask1 = (y == 1)
    fig.add_trace(go.Scatter3d(
        x=X[mask1, 0], y=X[mask1, 1], z=np.ones(sum(mask1)),  # Z=1
        mode='markers',
        name=f"{cfg.CLASS_NAMES[1]} (True=1)",
        marker=dict(color='blue', size=6, line=dict(width=2, color='white'))
    ))

    # --- 布局设置 ---
    fig.update_layout(
        title="3D Logistic Regression Surface (Sigmoid)",
        scene=dict(
            xaxis_title=data['names'][0],  # 特征1
            yaxis_title=data['names'][1],  # 特征2
            zaxis_title="Probability (P)",  # Z轴现在是概率了！

            # 视角调整
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1)),

            # Z轴范围锁定 0~1
            zaxis=dict(range=[0, 1.1]),
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    filename = "task3_probability_surface.html"
    fig.write_html(f"{cfg.OUTPUT_DIR}/{filename}")
    print(f"✅ 保存完毕: {cfg.OUTPUT_DIR}/{filename}")
    cfg.open_file(filename)