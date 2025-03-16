from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def iris_tree():
    iris = load_iris()
    X, Y = iris.data, iris.target
    feature_names, class_names = iris.feature_names, iris.target_names

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train Decision Tree
    dtree = DecisionTreeClassifier(random_state=42)
    dtree.fit(X_train, Y_train)

    # Model Evaluation
    accuracy = dtree.score(X_test, Y_test)
    print(f'Decision Tree Accuracy: {accuracy:.2f}')

    # Visualize Decision Tree
    plt.figure(figsize=(12, 8))
    plot_tree(dtree, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()

iris_tree()
