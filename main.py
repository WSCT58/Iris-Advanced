import argparse
from src.config import cfg
from src.data_manager import DataManager
from src.model_factory import ModelFactory
from src.evaluator import Evaluator
from src.visualizers import task1_2d, task2_3d, task3_3d, task4_3d
# ✅ 1. 导入新模块
from src.visualizers import extra_pca, extra_radar


def main():
    parser = argparse.ArgumentParser()
    # ✅ 2. 在 choices 里加上 'pca' 和 'radar'
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', '1', '2', '3', '4', '5', 'pca', 'radar'])
    args = parser.parse_args()

    dm = DataManager()

    # ... (Task 1,2,3,4,5 的代码保持不变) ...
    if args.task in ['all', '1']:
        data = dm.get_data('2d', binary=False)
        models = ModelFactory.get_9_classifiers()
        task1_2d.run(data, models)

    if args.task in ['all', '2']:
        data = dm.get_data('3d', binary=True)
        model = ModelFactory.get_model_for_3d_boundary()
        task2_3d.run(data, model)

    if args.task in ['all', '3']:
        data = dm.get_data('all', binary=True)  # 注意 Task 3 现在用全特征筛选
        model = ModelFactory.get_model_for_3d_prob()
        task3_3d.run(data, model)

    if args.task in ['all', '4']:
        data = dm.get_data('3d', binary=False)
        model = ModelFactory.get_model_for_hologram()
        task4_3d.run(data, model)

    if args.task in ['all', '5']:
        data = dm.get_data('all', binary=False)
        models = ModelFactory.get_9_classifiers()
        Evaluator().run(data, models)

    # ✅ 3. 加上这两个判断逻辑
    if args.task in ['all', 'pca']:
        data = dm.get_data('all', binary=False)
        extra_pca.run(data)

    if args.task in ['all', 'radar']:
        data = dm.get_data('all', binary=False)
        extra_radar.run(data)


if __name__ == "__main__":
    main()