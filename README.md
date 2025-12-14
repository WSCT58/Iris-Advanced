


# 🌸 Iris Advanced Analytics & Visualization System
## 鸢尾花数据集高级分析与全息可视化系统

> 一个集成了经典机器学习算法对比、交互式 3D 可视化、概率曲面分析以及 “上帝视角” 降维分析的综合数据科学项目。

本项目不仅仅是一个简单的分类任务，它采用 **Plotly** 和 **Matplotlib** 实现了工业级的数据可视化，重点展示了**全息体渲染（Volumetric Rendering）**、**概率热图（Probability Heatmaps）**和**雷达图（Radar Charts）**等高级技术。


## 📦 1. 项目功能模块 (Implemented Features)

本项目实现了从基础分类到高级全息可视化的完整工作流，具体功能如下：

### ✅ Task 1: 多分类器 2D 决策边界对比
*   **功能描述**：对比了 9 种主流算法（LR, SVM, KNN, Random Forest, Naive Bayes 等）在二维特征空间（花瓣长 / 宽）的表现。
*   **技术亮点**：绘制了清晰的决策区域背景，并正确标记了三分类数据点。

### ✅ Task 2: 3D 交互式决策边界
*   **功能描述**：在三维空间中构建二分类决策面，展示分类器如何 “切分” 空间。
*   **技术亮点**：使用 **Marching Cubes** 算法提取概率为 0.5 的等值面，生成可旋转、缩放的 3D 网格。

### ✅ Task 3: 3D 逻辑回归概率曲面
*   **功能描述**：教科书级的可视化。以 X/Y 轴为特征，Z 轴为概率值 (0.0~1.0)，展示逻辑回归的拟合过程。
*   **技术亮点**：绘制平滑的 Sigmoid 曲面，并在底部投影决策热图，直观展示概率梯度。

### ✅ Task 4.1: 3D 全息体渲染 (Holographic Map)
*   **功能描述**：结合了 “硬边界” 与 “软概率” 的高级视图，模拟全息投影效果。
*   **技术亮点**：
    *   **体渲染 (Volume Rendering)**：生成彩色概率云雾，颜色深浅代表模型置信度。
    *   **决策网格**：提取 Setosa 类别的金色分界网，明确划分空间。

### ✅ Task 4.2: 性能评估系统
*   **功能描述**：全方位的模型评分与排名。
*   **技术亮点**：
    *   **排行榜**：基于测试集准确率生成横向柱状图，直观展示 “版本之子”。
    *   **混淆矩阵**：自动绘制最佳模型的混淆矩阵，分析误判情况。
    *   **交叉验证**：计算 5 折 CV 分数，确保模型稳定性。

### 🔬 Task 4.3:Advanced Analysis (高级扩展)
*   **PCA 降维 ("上帝视角")**：使用 PCA 将 4 维特征压缩至 3 维，绘制特征向量箭头，展示特征对分类的贡献度。
*   **特征雷达图 ("植物 DNA")**：绘制特征雷达图，直观对比不同类别花卉的形态轮廓（平均特征值）。
*   **3D 交互式旋转**：PCA 视图完全支持在 3D 空间内的自由旋转与缩放，用户可以从任意角度观察数据簇的分离情况，获得比静态图表更直观的洞察。
---

---

## 📂 2. 项目结构

```text
Iris-Advanced-Project/
├── main.py                     # 🚀 主程序入口
├── src/
│   ├── config.py               # ⚙️ 全局配置 (颜色、分辨率、自动打开功能)
│   ├── data_manager.py         # 💾 数据清洗、标准化与切分
│   ├── model_factory.py        # 🤖 9种机器学习模型定义
│   ├── evaluator.py            # 📊 评估模块 (排行榜、混淆矩阵)
│   └── visualizers/            # 🎨 可视化模块
│       ├── task1_2d.py         # 2D 对比绘图
│       ├── task2_3d.py         # 3D 边界生成
│       ├── task3_3d.py         # 3D 概率曲面生成
│       ├── task4_3d.py         # 3D 全息图生成
│       ├── extra_pca.py        # PCA 分析
│       └── extra_radar.py      # 雷达图分析
└── results/                    # 📂 结果保存目录
```

---

## 🛠️ 3. 环境配置指南 (Virtual Environment)

为了保证项目运行的稳定性，建议创建独立的 Python 虚拟环境。

### 方法一：使用 venv (Python 自带)

**1. 创建虚拟环境**
在项目根目录下打开终端（Terminal / CMD），运行：
```bash
# Windows
python -m venv venv

# Mac / Linux
python3 -m venv venv
```

**2. 激活虚拟环境**
```bash
# Windows
.\venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```
*激活成功后，你的命令行前面会出现 `(venv)` 字样。*

**3. 安装依赖**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image plotly
```

### 方法二：使用 Conda (推荐)
```bash
# 1. 创建环境
conda create -n iris_env python=3.9

# 2. 激活环境
conda activate iris_env

# 3. 安装依赖
pip install numpy pandas matplotlib seaborn scikit-learn scikit-image plotly
```

---

## 🚀 4. 运行指南 (How to Run)

环境配置完成后，所有功能均通过统一入口 `main.py` 驱动。程序运行结束后，会自动调用系统默认浏览器或图片查看器打开生成的结果文件。

### 基础运行命令
```bash
# 运行 2D 决策边界对比
python main.py --task 1

# 运行 3D 交互式决策边界
python main.py --task 2

# 运行 3D 概率曲面
python main.py --task 3

# 运行 3D 全息概率云 + 决策网
python main.py --task 4

# 运行 性能排行榜 & 混淆矩阵
python main.py --task 5
```

### 高级分析命令
```bash
# 运行 PCA 降维分析 ("上帝视角")
python main.py --task pca

# 运行 特征雷达图分析 ("植物DNA")
python main.py --task radar
```

### 一键演示模式
```bash
# 依次运行所有任务
python main.py --task all
```

---

## ⚙️ 5. 参数配置 (Configuration)

本项目采用集中式配置管理，您无需修改底层代码逻辑，只需编辑 `src/config.py` 即可调整可视化风格与性能参数。

| 参数变量 (Variable) | 类型 | 默认值 | 作用描述 (Description) |
| :--- | :--- | :--- | :--- |
| **`COLORS`** | `List` | `['#FF5555', ...]` | **分类配色方案**。<br>修改十六进制代码可自定义 Setosa/Versicolor/Virginica 的颜色。 |
| **`RES_3D`** | `int` | `50` | **3D 网格密度**。<br>控制 Task 2 & 3 的平滑度。数值越高越精细，但渲染更慢。 |
| **`RES_VOL`** | `complex` | `100j` | **体渲染采样率**。<br>仅影响 Task 4。数值过高（如 200j）可能导致浏览器卡顿。 |
| **`FEATURES_2D`** | `tuple` | `(2, 3)` | **2D 特征索引**。<br>指定 Task 1 使用哪两个特征进行绘图（例如：花瓣长 vs 花瓣宽）。 |
| **`AUTO_OPEN`** | `bool` | `True` | **自动打开结果**。<br>设为 `True` 时，程序运行结束后会自动弹出图片或浏览器。 |

**✏️ 配置示例 (`src/config.py`):**

```python
# 调整配色为 "冷色调" 风格
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

# 提高 3D 渲染质量 (适合高性能电脑)
RES_3D = 100 
RES_VOL = 150j 
```