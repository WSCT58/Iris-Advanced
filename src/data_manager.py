import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import cfg  # ✅ 统一用这个！

class DataManager:
    def __init__(self):
        self.iris = load_iris()
        self.X_raw = self.iris.data
        self.y_raw = self.iris.target
        self.feature_names_raw = self.iris.feature_names

    def get_data(self, task_type='2d', binary=False):
        """
        根据任务类型返回处理好的数据
        task_type: '2d' (2特征) 或 '3d' (3特征) 或 'all' (4特征用于评估)
        """
        # 1. 特征筛选
        if task_type == '2d':
            feats = cfg.FEATURES_2D
        elif task_type == '3d':
            feats = cfg.FEATURES_3D
        else:
            feats = [0, 1, 2, 3]
            
        X = self.X_raw[:, feats]
        y = self.y_raw.copy()
        feat_names = [self.feature_names_raw[i] for i in feats]

        # 2. 二分类处理 (仅保留 Class 0 和 1)
        if binary:
            mask = y < 2
            X = X[mask]
            y = y[mask]

        # 3. 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.TEST_SIZE, 
            random_state=cfg.RANDOM_STATE, stratify=y
        )

        # 4. 标准化 (非常重要)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'names': feat_names, 'scaler': scaler,
            'X_full': scaler.transform(X), # 用于可视化画图的完整数据
            'y_full': y
        }