from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

digits = load_digits()

X, Y = digits.data, digits.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to find the best k
def find_best_k():
    best_k = 1
    best_acc = 0
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        knn.fit(X_train, Y_train)
        acc = accuracy_score(Y_test, knn.predict(X_test))

        if acc > best_acc:
            best_acc = acc
            best_k = k
    
    return best_k, best_acc

# Train KNN when the module is imported
best_k, best_accuracy = find_best_k()
knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)  # Store predictions globally

def get_knn_accuracy():
    return accuracy_score(Y_test, Y_pred)

def get_knn_classification_report():
    return classification_report(Y_test, Y_pred)

if __name__ == "__main__":
    plt.imshow(digits.images[0], cmap='gray')
    plt.title(f'Label: {digits.target[0]}')
    plt.show()

    print(f'Best k: {best_k} | Best Accuracy: {best_accuracy:.2f}')
    print(f'Final KNN Accuracy: {get_knn_accuracy():.2f}')
    print("Classification Report:\n", get_knn_classification_report())
