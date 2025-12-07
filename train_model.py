"""
Скрипт для обучения модели на датасете Titanic.
Сохраняет обученную модель в models/model.pkl
"""
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Titanic данные (упрощённые):
# Features: [Pclass, Sex (0=female, 1=male), Age, SibSp, Parch, Fare]
# Target: Survived (0/1)

# Примерные данные из Titanic dataset
X_train = np.array([
    [3, 1, 22, 1, 0, 7.25],    # male, 3rd class, young
    [1, 0, 38, 1, 0, 71.28],   # female, 1st class
    [3, 0, 26, 0, 0, 7.92],    # female, 3rd class
    [1, 0, 35, 1, 0, 53.10],   # female, 1st class
    [3, 1, 35, 0, 0, 8.05],    # male, 3rd class
    [1, 1, 54, 0, 0, 51.86],   # male, 1st class
    [3, 1, 2, 3, 1, 21.07],    # male child, 3rd class
    [3, 0, 27, 0, 2, 11.13],   # female, 3rd class
    [2, 0, 14, 1, 0, 30.07],   # female, 2nd class
    [3, 0, 4, 1, 1, 16.70],    # female child, 3rd class
    [1, 0, 58, 0, 0, 26.55],   # female, 1st class
    [3, 1, 20, 0, 0, 8.05],    # male, 3rd class
    [3, 1, 39, 1, 5, 31.27],   # male, 3rd class, big family
    [1, 0, 14, 1, 2, 120.00],  # female, 1st class
    [3, 0, 55, 0, 0, 7.79],    # female, 3rd class
    [2, 1, 31, 1, 0, 26.00],   # male, 2nd class
    [3, 1, 34, 0, 0, 6.50],    # male, 3rd class
    [2, 0, 34, 1, 1, 23.00],   # female, 2nd class
    [3, 1, 28, 0, 0, 7.90],    # male, 3rd class
    [1, 1, 30, 0, 0, 27.75],   # male, 1st class
])

y_train = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])

# Создаём pipeline со скейлером и логистической регрессией
model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Обучаем
model.fit(X_train, y_train)

# Проверяем
train_accuracy = model.score(X_train, y_train)
print(f"Train accuracy: {train_accuracy:.2%}")

# Тестовые предсказания
test_cases = [
    [1, 0, 29, 0, 0, 211.34],  # female, 1st class, high fare -> likely survived
    [3, 1, 25, 0, 0, 7.25],    # male, 3rd class, low fare -> likely died
]

for features in test_cases:
    pred = model.predict([features])[0]
    proba = model.predict_proba([features])[0]
    print(f"Features: {features}")
    print(f"  Prediction: {pred} (Survived)" if pred == 1 else f"  Prediction: {pred} (Died)")
    print(f"  Confidence: {max(proba):.2%}")

# Сохраняем модель
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved to models/model.pkl")

# Сохраняем информацию о фичах
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
print(f"Features order: {feature_names}")
print("Sex: 0=female, 1=male")
print("Pclass: 1=1st, 2=2nd, 3=3rd")

