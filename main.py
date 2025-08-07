import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AppConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent
    MODEL_NAME: str = 'churn_xgb_model.joblib'
    MODEL_PATH: Path = BASE_DIR / MODEL_NAME
    MODEL_COLUMNS: list[str] = [
        'кредитный_рейтинг', 'возраст', 'стаж_в_банке', 'баланс_депозита',
        'число_продуктов', 'есть_кредитка', 'активный_клиент', 'оценочная_зарплата',
        'balance_salary_ratio', 'tenure_by_age', 'balance_per_product',
        'is_balance_zero', 'город_Астана', 'город_Атырау', 'пол_Male'
    ]

config = AppConfig()

def load_model(path: Path):
    if not path.exists():
        logging.error(f"{path}")
        return None
    try:
        model = joblib.load(path)
        logging.info(f"{path}")
        return model
    except Exception as e:
        logging.error(f"{e}")
        return None

model = load_model(config.MODEL_PATH)

class ClientData(BaseModel):
    кредитный_рейтинг: int = Field(..., example=650, description="Кредитный рейтинг клиента")
    город: str = Field(..., example="Алматы", description="Город проживания клиента")
    пол: str = Field(..., example="Male", description="Пол клиента (Male/Female)")
    возраст: int = Field(..., example=42, description="Возраст клиента")
    стаж_в_банке: int = Field(..., example=5, description="Количество лет в банке")
    баланс_депозита: float = Field(..., example=120000.0, description="Баланс на депозите")
    число_продуктов: int = Field(..., example=1, description="Количество используемых продуктов")
    есть_кредитка: int = Field(..., example=1, description="Наличие кредитной карты (1 - да, 0 - нет)")
    активный_клиент: int = Field(..., example=0, description="Активность клиента (1 - да, 0 - нет)")
    оценочная_зарплата: float = Field(..., example=80000.0, description="Примерная годовая зарплата")
    balance_salary_ratio: float = Field(..., example=1.5, description="Отношение баланса к зарплате")
    tenure_by_age: float = Field(..., example=0.119, description="Отношение стажа к возрасту")
    balance_per_product: float = Field(..., example=120000.0, description="Средний баланс на продукт")
    is_balance_zero: int = Field(..., example=0, description="Флаг нулевого баланса (1 - да, 0 - нет)")

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int

app = FastAPI(
    title="Churn Prediction API",
    description="Сервис для предсказания оттока клиентов банка.",
    version="1.0.0"
)

@app.get("/", tags=["Health Check"])
def root():
    return {"status": "ok", "message": "Сервис предсказания оттока клиентов работает."}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: ClientData) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена или недоступна.")

    try:
        input_df = pd.DataFrame([data.dict()])

        input_df['город_Астана'] = (input_df['город'] == 'Астана').astype(int)
        input_df['город_Атырау'] = (input_df['город'] == 'Атырау').astype(int)
        input_df['пол_Male'] = (input_df['пол'] == 'Male').astype(int)
        input_df = input_df.drop(['город', 'пол'], axis=1)

        final_df = input_df.reindex(columns=config.MODEL_COLUMNS, fill_value=0)

        pred_proba = model.predict_proba(final_df)[0][1]
        prediction = model.predict(final_df)[0]

        return PredictionResponse(
            churn_probability=float(pred_proba),
            churn_prediction=int(prediction)
        )
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка: {e}")
