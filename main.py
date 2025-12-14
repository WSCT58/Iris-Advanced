import argparse
from config import cfg
from src.data_manager import DataManager
from src.model_factory import ModelFactory
from src.evaluator import Evaluator
from src.visualizers import task1_2d, task2_3d, task3_3d, task4_3d

def main():
    parser = argparse.ArgumentParser(description="Iris Advanced Project Framework")
    parser.add_argument('--task', type=str, default='all', 
                        choices=['all', '1', '2', '3', '4', '5'],
                        help='Choose which task to run (1-5 or all)')
    args = parser.parse_args()

    dm = DataManager()
    
    # --- Task 1: 2D 经典对比 ---
    if args.task in ['all', '1']:
        data = dm.get_data('2d', binary=False)
        models = ModelFactory.get_9_classifiers()
        task1_2d.run(data, models)

    # --- Task 2: 3D 边界 (二分类) ---
    if args.task in ['all', '2']:
        data = dm.get_data('3d', binary=True)
        model = ModelFactory.get_model_for_3d_boundary()
        task2_3d.run(data, model)

    # --- Task 3: 3D 概率层 (二分类) ---
    if args.task in ['all', '3']:
        data = dm.get_data('3d', binary=True)
        model = ModelFactory.get_model_for_3d_prob()
        task3_3d.run(data, model)

    # --- Task 4: 3D 全息体渲染 (多分类) ---
    if args.task in ['all', '4']:
        data = dm.get_data('3d', binary=False)
        model = ModelFactory.get_model_for_hologram()
        task4_3d.run(data, model)

    # --- Task 5: 性能评估 ---
    if args.task in ['all', '5']:
        data = dm.get_data('all', binary=False) # 使用所有4个特征进行评估
        models = ModelFactory.get_9_classifiers()
        Evaluator().run(data, models)

if __name__ == "__main__":
    main()