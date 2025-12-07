import os
import sys

import grpc

# Добавляем путь к protos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from protos import model_pb2, model_pb2_grpc

# Конфигурация
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = os.getenv("GRPC_PORT", "50051")


def get_channel():
    """Создание gRPC канала"""
    return grpc.insecure_channel(f'{GRPC_HOST}:{GRPC_PORT}')


def check_health():
    """Проверка здоровья сервиса"""
    with get_channel() as channel:
        stub = model_pb2_grpc.PredictionServiceStub(channel)
        response = stub.Health(model_pb2.HealthRequest())
        print(f"Health check:")
        print(f"  Status: {response.status}")
        print(f"  Model version: {response.model_version}")
        return response


def predict(features: list, description: str = ""):
    """
    Получение предсказания (Titanic survival).
    Features: [Pclass, Sex, Age, SibSp, Parch, Fare]
    """
    with get_channel() as channel:
        stub = model_pb2_grpc.PredictionServiceStub(channel)
        request = model_pb2.PredictRequest(features=features)
        response = stub.Predict(request)
        
        survival = "Survived" if response.prediction == 1 else "Died"
        print(f"Prediction{f' ({description})' if description else ''}:")
        print(f"  Features: {features}")
        print(f"  Result: {response.prediction} ({survival})")
        print(f"  Confidence: {response.confidence:.2%}")
        print(f"  Model version: {response.model_version}")
        return response


def main():
    """Тестирование сервиса Titanic prediction"""
    print("=" * 60)
    print("Testing Titanic Survival Prediction Service")
    print("Features: [Pclass, Sex, Age, SibSp, Parch, Fare]")
    print("Sex: 0=female, 1=male | Pclass: 1=1st, 2=2nd, 3=3rd")
    print("=" * 60)
    
    # Тест health
    print("\n1. Health check:")
    try:
        check_health()
    except grpc.RpcError as e:
        print(f"Error: {e.code()} - {e.details()}")
        return
    
    # Тестовые случаи
    test_cases = [
        ([1, 0, 29, 0, 0, 211.34], "Rose (1st class woman)"),
        ([3, 1, 25, 0, 0, 7.25], "Jack (3rd class man)"),
        ([1, 1, 4, 1, 2, 120.00], "Child (1st class)"),
        ([3, 0, 55, 0, 0, 7.79], "Elderly woman (3rd class)"),
        ([2, 1, 35, 1, 0, 26.00], "Man (2nd class)"),
    ]
    
    print("\n2. Prediction tests:")
    for features, description in test_cases:
        print()
        try:
            predict(features, description)
        except grpc.RpcError as e:
            print(f"  Error: {e.code()} - {e.details()}")
    
    # Тест невалидных данных
    print("\n3. Invalid input test:")
    try:
        predict([1, 2, 3], "Wrong number of features")
    except grpc.RpcError as e:
        print(f"  Expected error: {e.code()}")
        print(f"  Details: {e.details()}")


if __name__ == "__main__":
    main()
