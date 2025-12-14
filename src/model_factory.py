from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class ModelFactory:
    @staticmethod
    def get_9_classifiers():
        """返回 Task 1 所需的9种经典分类器"""
        return {
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(),
            "Linear SVM": SVC(kernel="linear", probability=True),
            "RBF SVM": SVC(kernel="rbf", probability=True),
            "Naive Bayes": GaussianNB(),
            "LDA": LinearDiscriminantAnalysis(),
            "Decision Tree": DecisionTreeClassifier(max_depth=5),
            "Random Forest": RandomForestClassifier(n_estimators=50),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=50)
        }

    @staticmethod
    def get_model_for_3d_boundary():
        """Task 2: RBF SVM 适合画平滑边界"""
        return SVC(kernel='rbf', probability=True)

    @staticmethod
    def get_model_for_3d_prob():
        """Task 3: 逻辑回归适合展示概率梯度"""
        return LogisticRegression()

    @staticmethod
    def get_model_for_hologram():
        """Task 4: RBF SVM 产生的体素云雾效果最好"""
        return SVC(kernel='rbf', probability=True)