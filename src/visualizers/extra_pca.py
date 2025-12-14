import plotly.graph_objects as go
from sklearn.decomposition import PCA
from src.config import cfg


def run(data):
    print("🎨 [Extra] 生成 PCA 3D 降维投影...")

    # 1. 获取全部4维数据 (已标准化)
    X, y = data['X_full'], data['y_full']

    # 2. PCA 降维 (4维 -> 3维)
    # 这一步是把 4 个特征压缩成 3 个"主成分"
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # 计算保留了多少信息量 (解释方差比)
    explained_var = pca.explained_variance_ratio_
    total_var = sum(explained_var) * 100

    fig = go.Figure()

    # 3. 绘制数据点
    for cls in [0, 1, 2]:
        mask = (y == cls)
        fig.add_trace(go.Scatter3d(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            z=X_pca[mask, 2],
            mode='markers',
            name=f"{cfg.CLASS_NAMES[cls]}",
            marker=dict(
                size=6,
                color=cfg.COLORS[cls],
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            hovertemplate=f"<b>{cfg.CLASS_NAMES[cls]}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}"
        ))

    # 4. 添加特征向量 (箭头)
    # 这部分展示了原始特征 (如花瓣长) 在这个新空间里的方向
    loadings = pca.components_.T * 3  # 放大系数
    features = ['Sepal Len', 'Sepal Wid', 'Petal Len', 'Petal Wid']

    for i, feature in enumerate(features):
        fig.add_trace(go.Scatter3d(
            x=[0, loadings[i, 0]],
            y=[0, loadings[i, 1]],
            z=[0, loadings[i, 2]],
            mode='lines+text',
            text=[None, feature],
            textposition="top center",
            line=dict(color='black', width=5),  # 加粗箭头
            name=f"Vector: {feature}"
        ))
        # 箭头头部
        fig.add_trace(go.Scatter3d(
            x=[loadings[i, 0]], y=[loadings[i, 1]], z=[loadings[i, 2]],
            mode='markers', marker=dict(size=5, color='black'), showlegend=False
        ))

    fig.update_layout(
        title=f"PCA 'God View' (Retains {total_var:.1f}% Info)",
        scene=dict(
            xaxis_title=f"PC1 ({explained_var[0]:.1%})",
            yaxis_title=f"PC2 ({explained_var[1]:.1%})",
            zaxis_title=f"PC3 ({explained_var[2]:.1%})",
            aspectmode='cube'
        )
    )

    filename = "extra_pca_3d.html"
    fig.write_html(f"{cfg.OUTPUT_DIR}/{filename}")
    print(f"✅ 保存完毕: {cfg.OUTPUT_DIR}/{filename}")
    cfg.open_file(filename)