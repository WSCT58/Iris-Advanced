import plotly.graph_objects as go
import pandas as pd
from src.config import cfg


def run(data):
    print("🎨 [Extra] 生成特征雷达图 (Flower DNA)...")

    X, y = data['X_full'], data['y_full']
    # 构造 DataFrame 以便计算均值
    df = pd.DataFrame(X, columns=['Sepal Len', 'Sepal Wid', 'Petal Len', 'Petal Wid'])
    df['target'] = y

    # 计算每一类的特征均值
    df_mean = df.groupby('target').mean()
    categories = list(df.columns[:-1])

    fig = go.Figure()

    for cls in [0, 1, 2]:
        # 数据闭环技巧：把第一个点加到列表末尾，让线条连成圈
        values = df_mean.iloc[cls].values.tolist()
        values += values[:1]
        cats = categories + [categories[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill='toself',
            name=cfg.CLASS_NAMES[cls],
            line=dict(color=cfg.COLORS[cls], width=3),
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[-2, 2])  # 范围根据标准化数据调整
        ),
        title="Average Feature Profile (Shape DNA)",
        template="plotly_white"
    )

    filename = "extra_radar_chart.html"
    fig.write_html(f"{cfg.OUTPUT_DIR}/{filename}")
    print(f"✅ 保存完毕: {cfg.OUTPUT_DIR}/{filename}")
    cfg.open_file(filename)