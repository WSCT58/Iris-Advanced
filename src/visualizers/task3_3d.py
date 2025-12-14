import plotly.graph_objects as go
import numpy as np
from config import cfg

def run(data, model):
    print("🎨 [Task 3] 生成 3D 连续概率热图 (Volume Heatmap)...")
    
    # 1. 获取二分类数据
    X, y = data['X_full'], data['y_full']
    model.fit(X, y)

    # 2. 生成高密度体素网格 (为了让雾气看起来细腻，分辨率设高一点)
    res = 35 
    pad = 0.5
    
    # 计算包围盒
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    z_min, z_max = X[:, 2].min()-pad, X[:, 2].max()+pad
    
    gx, gy, gz = np.mgrid[x_min:x_max:complex(0, res), 
                          y_min:y_max:complex(0, res), 
                          z_min:z_max:complex(0, res)]
    
    # 3. 预测每个体素点的概率 (Class 1 的概率)
    # Logistic Regression 输出的是 0.0 到 1.0 的平滑数值
    probs = model.predict_proba(np.c_[gx.ravel(), gy.ravel(), gz.ravel()])[:, 1]
    
    fig = go.Figure()

    # --- 核心修改：使用 Volume 展示连续概率场 ---
    fig.add_trace(go.Volume(
        x=gx.flatten(),
        y=gy.flatten(),
        z=gz.flatten(),
        value=probs, # 这里传入的是具体的概率值，而不是类别
        
        # 概率范围 0.0 ~ 1.0
        isomin=0.0,
        isomax=1.0,
        
        # 设置透明度：让概率低(0.5左右)的地方透明，概率极端的地方不透明
        opacity=0.1, 
        
        # 采样层数：越多越平滑
        surface_count=20, 
        
        # 颜色映射：RdBu (红-白-蓝)
        # 红色 = Class 0 (Prob -> 0)
        # 蓝色 = Class 1 (Prob -> 1)
        colorscale='RdBu',
        
        # 显示颜色条，告诉用户哪个颜色对应多少概率
        colorbar=dict(title="Probability of Class 1"),
        name='Probability Cloud'
    ))

    # 4. 绘制真实数据点 (悬浮在概率云中)
    for cls in [0, 1]:
        mask = y == cls
        # 颜色对应：0用红色系，1用蓝色系，与热图呼应
        point_color = 'red' if cls == 0 else 'blue'
        
        fig.add_trace(go.Scatter3d(
            x=X[mask, 0], y=X[mask, 1], z=X[mask, 2],
            mode='markers',
            name=f"{cfg.CLASS_NAMES[cls]} Data",
            marker=dict(
                color=point_color, 
                size=6, 
                line=dict(width=2, color='white'),
                opacity=0.9
            )
        ))

    fig.update_layout(
        title="3D Continuous Probability Density (Logicist Regression)",
        scene=dict(
            xaxis_title=data['names'][0],
            yaxis_title=data['names'][1],
            zaxis_title=data['names'][2],
            aspectmode='data'
        )
    )

    save_path = f"{cfg.OUTPUT_DIR}/task3_probability_heatmap.html"
    fig.write_html(save_path)
    print(f"✅ 保存完毕: {save_path}")