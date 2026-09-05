"""시간순 분리로 학습/평가하고 모델을 저장한다. 셔플 없음 — 과거로 미래를 맞추는 걸 검증해야 하므로."""
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from data import load_matches
from features import build_features, FEATURE_NAMES
from fetch_data import COMPLETED_SEASONS

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.joblib"
TEST_FRACTION = 0.2


def time_based_split(df):
    """날짜 기준으로 분리하고, 동일 킥오프 시각 경기가 경계에 걸리지 않는지 명시적으로 보장한다."""
    split_idx = int(len(df) * (1 - TEST_FRACTION))
    split_date = df.iloc[split_idx]["date"]
    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]
    assert train_df["date"].max() < test_df["date"].min(), "train/test 경계에 리키지 발생"
    return train_df, test_df


def main() -> None:
    matches = load_matches(seasons=COMPLETED_SEASONS)  # 진행 중 시즌의 소량 표본은 학습에서 제외
    feats = build_features(matches)  # 이미 date 오름차순 정렬됨

    train_df, test_df = time_based_split(feats)
    print(f"train: {len(train_df)} matches ({train_df['date'].min()} ~ {train_df['date'].max()})")
    print(f"test:  {len(test_df)} matches ({test_df['date'].min()} ~ {test_df['date'].max()})")

    X_train, y_train = train_df[FEATURE_NAMES], train_df["result"]
    X_test, y_test = test_df[FEATURE_NAMES], test_df["result"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)
    preds = model.classes_[np.argmax(proba, axis=1)]

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, proba, labels=model.classes_)
    majority_baseline = y_train.value_counts(normalize=True).max()

    print(f"accuracy: {acc:.3f}  (train-set majority-class baseline: {majority_baseline:.3f})")
    print(f"log_loss: {ll:.3f}")

    # 학습할 때마다 새로 발급 — predictions 테이블에서 "이 예측이 어느 학습 결과로
    # 나왔는지" 구분하는 키가 된다. 마이크로초까지 넣어 같은 초 안에 두 번 재학습해도
    # (예: 재시도, 병렬 CI 잡) 충돌하지 않게 한다.
    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": FEATURE_NAMES, "classes": list(model.classes_), "model_version": model_version},
        MODEL_PATH,
    )
    print(f"saved model -> {MODEL_PATH} (model_version={model_version})")


if __name__ == "__main__":
    main()
