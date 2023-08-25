import os
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

def load_dataset(directory_path):
    combined_data = pd.DataFrame()

    # 디렉토리 내 파일 탐색
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, data], ignore_index=True)

    return combined_data


def preprocess_data(data):
    # 결측값이 있는 행 삭제
    data = data.dropna()

    # 별도의 형상 및 라벨
    labels = data[' Label']
    features = data.drop(' Label', axis=1)

    # 범주형 레이블을 숫자로 변환
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # 특이치 처리에 동기화 적용
    winsorized_features = features.apply(lambda x: winsorize(x, limits=[0.01, 0.01]), axis=0)

    # StandardScaler를 사용하여 기능 확장
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(winsorized_features)

    return scaled_features, labels


def build_model(input_shape, num_classes):
    # Sequential 모델 생성
    model = Sequential([
        # 첫 번째 레이어: 64개의 뉴런과 ReLU 활성화 함수를 가지는 Dense 레이어
        Dense(64, activation='relu', input_shape=input_shape),
        # Dropout 레이어를 사용하여 과적합을 방지합니다. 학습 중에 20%의 뉴런이 무작위로 비활성화됩니다.
        Dropout(0.2),
        # 두 번째 레이어: 64개의 뉴런과 ReLU 활성화 함수를 가지는 Dense 레이어
        Dense(64, activation='relu'),
        # Dropout 레이어 (학습 중에 20%의 뉴런이 무작위로 비활성화됩니다.)
        Dropout(0.2),
        # 세 번째 레이어: 다중 클래스 분류를 위해 num_classes 개의 뉴런과 softmax 활성화 함수를 가지는 Dense 레이어
        Dense(num_classes, activation='softmax')
    ])

    return model



def main():
    # 데이터셋 경로
    directory_path = "C:/Users/lhw93/Desktop/MachineLearningCSV/MachineLearningCVE"

    # 데이터셋 불러오기
    combined_data = load_dataset(directory_path)

    # 데이터 전처리
    features, labels = preprocess_data(combined_data)

    # 클래스 개수 계산
    num_classes = len(set(labels))
    print(len(set(labels)))
    # 훈련 및 테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # 모델 구축
    input_shape = X_train.shape[1:]
    model = build_model(input_shape, num_classes)

    # 모델 컴파일
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 조기 종료 콜백 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    # 모델 훈련
    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=10, callbacks=[early_stopping])

    # 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save('cicids2017_model')

if __name__ == '__main__':
    main()
