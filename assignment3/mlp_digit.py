from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


digits = load_digits()
X, Y = digits.data, digits.target 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train MLP with optimized hyperparameters
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=1000,
    batch_size=32,
    momentum=0.9,
    random_state=42
)

mlp.fit(X_train, Y_train) 
Y_pred = mlp.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Final MLP Accuracy: {accuracy:.2f}')
print("Classification Report:\n", classification_report(Y_test, Y_pred))
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred))
