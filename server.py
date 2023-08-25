# 필요한 라이브러리 및 모듈 import
import flwr as fl  # 연합학습을 위한 Flower 라이브러리
import tensorflow as tf  # 딥러닝 모델 생성 및 학습을 위한 라이브러리

# Flower 서버를 정의하는 클래스
class Server(fl.server.Server):
    def __init__(self, model, num_classes):
        self.model = model  # 전역 모델
        self.num_classes = num_classes  # 클래스 개수 설정
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 모델 컴파일

    # 전역 모델의 가중치를 가져오는 메소드
    def get_parameters(self):
        return self.model.get_weights()

    # 전역 모델의 가중치를 설정하는 메소드
    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    # 클라이언트 관리자 생성
    def client_manager(self):
        return fl.server.SimpleClientManager()

    # FL 라운드를 수행하는 메소드
    def fit(self, parameters, config):
        self.model.set_weights(parameters)  # 전달받은 파라미터로 모델 파라미터 업데이트
        global_weights = parameters  # 초기 전역 가중치 설정
        global_loss = 0.0  # 전역 손실 초기화
        global_samples = 0  # 전체 샘플 수 초기화
        for _ in range(config["num_rounds"]):  # 설정된 라운드 수만큼 반복
            # 클라이언트의 업데이트를 수집하고 평가 결과를 수집하여 전역 모델 업데이트
            weights, num_samples, _ = self.client_fit(global_weights)  # 클라이언트 업데이트 수집
            global_weights = weights  # 업데이트된 가중치 저장
            loss, _, _ = self.client_evaluate(global_weights)  # 클라이언트 평가 결과 수집
            global_loss += loss * num_samples  # 손실 누적
            global_samples += num_samples  # 샘플 수 누적
        global_loss /= global_samples  # 평균 손실 계산
        return global_weights, global_samples, {"loss": global_loss}  # 업데이트된 가중치, 샘플 수, 평균 손실 반환

    # 전역 모델 평가를 수행하는 메소드
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)  # 전달받은 파라미터로 모델 파라미터 업데이트
        X_test, y_test = self.test_data  # 검증 데이터 추출
        loss, accuracy = self.model.evaluate(X_test, y_test)  # 모델 평가
        return loss, len(X_test), {"accuracy": accuracy}  # 손실, 검증 데이터 개수, 정확도 반환

# main 함수 정의
def main():
    # 기존에 학습한 모델 파일 경로와 클래스 개수 설정
    model_path = 'cicids2017_model'
    num_classes = 10

    # 기존에 학습한 모델 불러오기
    global_model = tf.keras.models.load_model(model_path)

    # Flower 서버 생성 및 실행
    server = Server(global_model, num_classes)
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),  # 라운드 수 설정
        client_manager=server.client_manager()  # 클라이언트 관리자 설정
    )

# main 함수 호출
if __name__ == '__main__':
    main()
