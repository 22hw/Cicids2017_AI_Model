# 필요한 라이브러리 및 모듈 import
import os  # 파일 및 디렉토리 작업을 위한 라이브러리
import pandas as pd  # 데이터 조작을 위한 라이브러리
import flwr as fl  # 연합학습을 위한 Flower 라이브러리
import numpy as np  # 수치 연산을 위한 라이브러리
import tensorflow as tf  # 딥러닝 모델 생성 및 학습을 위한 라이브러리
from keras.layers import Dense, Dropout  # Keras를 활용한 딥러닝 모델 구성을 위한 모듈
from keras.models import Sequential  # 딥러닝 모델을 위한 Sequential 모델 클래스
from matplotlib import pyplot as plt  # 데이터 시각화를 위한 라이브러리
from scipy.stats.mstats import winsorize  # 통계 관련 함수를 포함한 모듈
from sklearn.model_selection import train_test_split  # 데이터 분할을 위한 모듈
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 데이터 전처리를 위한 모듈

# 데이터셋 로드
def load_dataset(directory_path):
    combined_data = pd.DataFrame()  # 빈 데이터프레임 생성
    for filename in os.listdir(directory_path):  # 지정된 디렉토리 내 파일 목록 순회
        if filename.endswith("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"):  # 특정 파일명 패턴 확인
            file_path = os.path.join(directory_path, filename)  # 파일 경로 생성
            data = pd.read_csv(file_path)  # CSV 파일 읽어 데이터프레임으로 로드
            combined_data = pd.concat([combined_data, data], ignore_index=True)  # 데이터 병합
    return combined_data  # 병합된 데이터프레임 반환

# 데이터 전처리
def preprocess_data(data):
    data = data.dropna()  # 결측치 제거
    labels = data[' Label']  # 레이블 추출
    features = data.drop(' Label', axis=1)  # 레이블 열을 제외한 특성 추출
    label_encoder = LabelEncoder()  # 레이블을 숫자로 변환하기 위한 LabelEncoder 객체 생성
    labels = label_encoder.fit_transform(labels)  # 범주형 레이블을 숫자로 변환
    winsorized_features = features.apply(lambda x: winsorize(x, limits=[0.01, 0.01]), axis=0)  # 데이터 정규화
    scaler = StandardScaler()  # 데이터 스케일링을 위한 StandardScaler 객체 생성
    scaled_features = scaler.fit_transform(winsorized_features)  # 데이터 스케일링 수행
    return scaled_features, labels  # 스케일링된 특성과 레이블 반환

# 모델 생성
def build_model(input_shape, num_classes):
    model = Sequential([  # Sequential 모델 생성
        Dense(64, activation='relu', input_shape=input_shape),  # 입력 레이어 (64개 뉴런, ReLU 활성화 함수)
        Dropout(0.2),  # 드롭아웃 레이어 (과적합 방지를 위한 드롭아웃)
        Dense(64, activation='relu'),  # 은닉 레이어 (64개 뉴런, ReLU 활성화 함수)
        Dropout(0.2),  # 드롭아웃 레이어
        Dense(num_classes, activation='softmax')  # 출력 레이어 (클래스 수만큼 뉴런, 소프트맥스 활성화 함수)
    ])
    return model  # 생성된 모델 반환

# Client 클래스 정의 (Flower 클라이언트)
class Client(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model  # 전달받은 모델
        self.X_train = X_train  # 훈련용 특성 데이터
        self.y_train = y_train  # 훈련용 레이블 데이터
        self.X_test = X_test  # 검증용 특성 데이터
        self.y_test = y_test  # 검증용 레이블 데이터
        self.train_data = (X_train, y_train)  # 훈련 데이터 튜플
        self.test_data = (X_test, y_test)  # 검증 데이터 튜플
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}  # 학습 및 검증 결과 기록용 딕셔너리

    def get_parameters(self, config=None):
        return self.model.get_weights()  # 모델 파라미터 반환

    def fit(self, parameters, config):
        self.model.set_weights(parameters)  # 전달받은 파라미터로 모델 파라미터 업데이트
        X_train, y_train = self.train_data  # 훈련 데이터 추출
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # 모델 컴파일
        history = self.model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(self.X_test, self.y_test))  # 모델 학습
        self.history['loss'].extend(history.history['loss'])  # 훈련 데이터의 손실 기록 저장
        self.history['accuracy'].extend(history.history['accuracy'])  # 훈련 데이터의 정확도 기록 저장
        self.history['val_loss'].extend(history.history['val_loss'])  # 검증 데이터의 손실 기록 저장
        self.history['val_accuracy'].extend(history.history['val_accuracy'])  # 검증 데이터의 정확도 기록 저장
        return self.model.get_weights(), len(X_train), {}  # 업데이트된 파라미터, 훈련 데이터 개수, 빈 딕셔너리 반환

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)  # 전달받은 파라미터로 모델 파라미터 업데이트
        X_test, y_test = self.test_data  # 검증 데이터 추출
        loss, accuracy = self.model.evaluate(X_test, y_test)  # 모델 평가
        return loss, len(X_test), {"accuracy": accuracy}  # 손실, 검증 데이터 개수, 정확도 반환

# main 함수 정의
def main():
    # 데이터셋 경로
    directory_path = "C:/Users/lhw93/Desktop/MachineLearningCSV/MachineLearningCVE"
    combined_data = load_dataset(directory_path)  # 데이터셋 로드
    features, labels = preprocess_data(combined_data)  # 데이터 전처리
    num_classes = len(set(labels))  # 레이블 클래스 수 계산
    print(len(set(labels)))  # 레이블 클래스 수 출력
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)  # 데이터 분할

    global_model = build_model(X_train.shape[1:], num_classes)  # 전역 모델 초기화

    # Flower 클라이언트 생성 및 실행
    client = Client(global_model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

    # 학습 및 검증 결과 시각화
    plt.plot(client.history['accuracy'])
    plt.plot(client.history['val_accuracy'])
    plt.title('Client1_Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(client.history['loss'])
    plt.plot(client.history['val_loss'])
    plt.title('Client1_Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# main 함수 호출
if __name__ == '__main__':
    main()
