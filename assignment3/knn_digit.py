from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt


digits = load_digits()

# Displaying the first digit image
plt.imshow(digits.images[0], cmap='gray')
plt.title(f'Label: {digits.target[0]}')
plt.show()

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


best_k, best_accuracy = find_best_k()
print(f'Best k: {best_k} | Best Accuracy: {best_accuracy:.2f}')

# Train final KNN model with optimal k
knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance', metric='euclidean')
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

# Evaluate Model
print(f'Final KNN Accuracy: {accuracy_score(Y_test, Y_pred):.2f}')
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))

# Visualize some predictions
# fig, axes = plt.subplots(3, 5, figsize=(10, 6))
# axes = axes.ravel()
# for i in range(15):
    # axes[i].imshow(X_test[i].reshape(8, 8), cmap='gray')
    # axes[i].set_title(f'Pred: {Y_pred[i]} (Actual: {Y_test[i]})')
    # axes[i].axis('off')
# plt.tight_layout()
# plt.show()
