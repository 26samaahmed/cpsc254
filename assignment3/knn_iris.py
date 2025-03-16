from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


iris = datasets.load_iris()
X, Y = iris.data, iris.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Testing different k values to find the best one
def find_best_k():
    best_k = 1
    best_acc = 0
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        
        if acc > best_acc:
            best_acc = acc
            best_k = k
    
    return best_k, best_acc


best_k, best_accuracy = find_best_k()
print(f'Best k: {best_k} | Best Accuracy: {best_accuracy:.2f}')

# Training final KNN model with best k
knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

print(f'Final KNN Accuracy: {accuracy_score(Y_test, Y_pred):.2f}')
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
