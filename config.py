import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class AppConfig:
    # --- 基础路径 ---
    OUTPUT_DIR: str = "results"
    
    # --- 数据参数 ---
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.3
    
    # --- 特征选择 (基于索引) ---
    # 0:Sepal Length, 1:Sepal Width, 2:Petal Length, 3:Petal Width
    FEATURES_2D: List[int] = field(default_factory=lambda: [2, 3]) 
    FEATURES_3D: List[int] = field(default_factory=lambda: [1, 2, 3]) 
    
    # --- 视觉风格 ---
    # 红(Setosa), 绿(Versicolor), 蓝(Virginica)
    COLORS: List[str] = field(default_factory=lambda: ['#FF6B6B', '#4ECDC4', '#45B7D1'])
    CLASS_NAMES: List[str] = field(default_factory=lambda: ['Setosa', 'Versicolor', 'Virginica'])
    
    # 背景色 (浅色版，用于Task 1背景)
    BG_COLORS: List[str] = field(default_factory=lambda: ['#FFCCCC', '#CCFFCC', '#CCCCFF'])
    
    # --- 分辨率 ---
    RES_2D: int = 200
    RES_3D: int = 40   # Marching Cubes 网格密度
    RES_VOL: int = 30  # 体渲染网格密度

    def __post_init__(self):
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)

# 实例化单例配置
cfg = AppConfig()