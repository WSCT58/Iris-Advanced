import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from config import cfg

class Evaluator:
    def run(self, data, models):
        print("\n📊 [Task 5] 开始模型性能评估...")
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
        X_full, y_full = data['X_full'], data['y_full'] # 用于CV

        results = []
        
        print(f"{'Algorithm':<25} | {'Test Acc':<10} | {'CV Mean':<10}")
        print("-" * 50)
        
        for name, model in models.items():
            # 1. 训练与测试
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            # 2. 交叉验证
            cv_scores = cross_val_score(model, X_full, y_full, cv=5)
            cv_mean = cv_scores.mean()
            
            print(f"{name:<25} | {acc:.4f}     | {cv_mean:.4f}")
            results.append({'Model': name, 'Test Acc': acc, 'CV Mean': cv_mean})
        
        # 保存到CSV
        df = pd.DataFrame(results)
        df.to_csv(f"{cfg.OUTPUT_DIR}/performance_metrics.csv", index=False)
        print(f"\n✅ 评估报告已保存至 {cfg.OUTPUT_DIR}/performance_metrics.csv")