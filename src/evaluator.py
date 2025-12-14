import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from src.config import cfg


class Evaluator:
    def run(self, data, models):
        print("\n📊 [Task 5] 开始模型性能评估...")
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        X_full, y_full = data['X_full'], data['y_full']

        results = []

        # 1. 计算所有模型的准确率
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            cv_scores = cross_val_score(model, X_full, y_full, cv=5)
            cv_mean = cv_scores.mean()

            results.append({'Model': name, 'Test Accuracy': acc, 'CV Mean': cv_mean})

        df = pd.DataFrame(results)
        df.to_csv(f"{cfg.OUTPUT_DIR}/performance_metrics.csv", index=False)
        print("✅ 基础评估完成")

        # 2. 绘制排行榜
        self.plot_leaderboard(df)

        # 3. 绘制最佳模型的混淆矩阵 (新增功能)
        best_row = df.sort_values(by='Test Accuracy', ascending=False).iloc[0]
        best_name = best_row['Model']
        best_model = models[best_name]  # 取出那个最好的模型对象

        # 重新训练一遍以确保是最佳状态
        best_model.fit(X_train, y_train)
        self.plot_confusion_matrix(best_model, X_test, y_test, best_name)

    def plot_leaderboard(self, df):
        """绘制排行榜"""
        df_sorted = df.sort_values(by='Test Accuracy', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(x='Test Accuracy', y='Model', data=df_sorted, palette='viridis')

        for i, v in enumerate(df_sorted['Test Accuracy']):
            ax.text(v + 0.01, i, f"{v:.2%}", color='black', va='center', fontweight='bold')

        plt.title('Classifier Leaderboard', fontsize=15, fontweight='bold')
        plt.xlabel('Accuracy')
        plt.xlim(0, 1.15)
        plt.tight_layout()

        filename = "model_leaderboard.png"
        plt.savefig(f"{cfg.OUTPUT_DIR}/{filename}", dpi=150)
        cfg.open_file(filename)  # 自动打开

    def plot_confusion_matrix(self, model, X_test, y_test, model_name):
        """绘制混淆矩阵"""
        print(f"🔍 绘制 {model_name} 的混淆矩阵...")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=cfg.CLASS_NAMES,
                    yticklabels=cfg.CLASS_NAMES)

        plt.title(f'Confusion Matrix: {model_name}', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        filename = "best_model_confusion_matrix.png"
        plt.savefig(f"{cfg.OUTPUT_DIR}/{filename}", dpi=150)
        cfg.open_file(filename)  # 自动打开