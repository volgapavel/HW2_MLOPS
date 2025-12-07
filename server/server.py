import os
import sys
import pickle
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
import numpy as np

# Добавляем путь к protos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from protos import model_pb2, model_pb2_grpc

# Конфигурация из переменных окружения
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1.0.0")
GRPC_PORT = os.getenv("GRPC_PORT", "50051")


class PredictionServicer(model_pb2_grpc.PredictionServiceServicer):
    """gRPC сервис для ML модели (Titanic survival prediction)"""
    
    def __init__(self):
        self.model_version = MODEL_VERSION
        self.model = self._load_model()
        self.feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
        print(f"Model service initialized", flush=True)
        print(f"  Version: {self.model_version}", flush=True)
        print(f"  Model path: {MODEL_PATH}", flush=True)
        print(f"  Features: {self.feature_names}", flush=True)
    
    def _load_model(self):
        """Загрузка модели из pickle файла"""
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded successfully from {MODEL_PATH}", flush=True)
            return model
        except FileNotFoundError:
            print(f"WARNING: Model file not found at {MODEL_PATH}", flush=True)
            print("Using dummy model (returns random predictions)", flush=True)
            return None
        except Exception as e:
            print(f"ERROR loading model: {e}", flush=True)
            return None
    
    def Health(self, request, context):
        """Проверка здоровья сервиса"""
        status = "ok" if self.model is not None else "degraded"
        return model_pb2.HealthResponse(
            status=status,
            model_version=self.model_version
        )
    
    def Predict(self, request, context):
        """
        Получение предсказания модели.
        Ожидает 6 фич: [Pclass, Sex, Age, SibSp, Parch, Fare]
        """
        features = list(request.features)
        
        # Валидация
        if len(features) != 6:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Expected 6 features [Pclass, Sex, Age, SibSp, Parch, Fare], "
                f"got {len(features)}"
            )
            return model_pb2.PredictResponse()
        
        features_array = np.array(features).reshape(1, -1)
        
        if self.model is None:
            prediction = int(np.random.randint(0, 2))
            confidence = float(np.random.uniform(0.5, 0.9))
        else:
            prediction = int(self.model.predict(features_array)[0])
            probabilities = self.model.predict_proba(features_array)[0]
            confidence = float(max(probabilities))
        
        return model_pb2.PredictResponse(
            prediction=prediction,
            confidence=confidence,
            model_version=self.model_version
        )


def serve():
    """Запуск gRPC сервера с reflection"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_PredictionServiceServicer_to_server(PredictionServicer(), server)
    
    # Включаем reflection для grpcurl
    SERVICE_NAMES = (
        model_pb2.DESCRIPTOR.services_by_name['PredictionService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    server.add_insecure_port(f'[::]:{GRPC_PORT}')
    server.start()
    
    print(f"gRPC server started on port {GRPC_PORT}", flush=True)
    print(f"Reflection enabled for grpcurl", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
