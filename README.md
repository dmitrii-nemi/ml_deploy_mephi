# Сервис прогнозирования дефолта по кредитным картам

Production-like Flask-сервис для прогнозирования дефолта по кредитной карте в следующем месяце. Проект обучает и обслуживает модели бинарной классификации на датасете UCI Default of Credit Card Clients.

Docker-образ: `batonhleba/mephi_ml:latest`

## Возможности

- Две версии модели для A/B-тестирования:
  - `v1`: Logistic Regression;
  - `v2`: Random Forest.
- Feature engineering встроен в сохранённый `sklearn Pipeline`.
- Артефакты моделей сохранены через `joblib`.
- Flask API с эндпоинтами для проверки состояния, одиночного прогноза и batch-прогноза.
- Стабильное распределение трафика 50/50 между `v1` и `v2`, если `model_version` не передан в запросе.
- Логирование запросов в формате JSONL.
- Dockerfile, Docker Compose, Makefile и GitHub Actions CI.
- Документация по архитектуре, MLOps-концептам и плану A/B-теста.

## Структура проекта

```text
.
├── .github/workflows/ci.yml
├── data/raw/UCI_Credit_Card.csv
├── docs/
│   ├── AB_TEST_PLAN.md
│   └── ARCHITECTURE.md
├── models/
│   ├── credit_default_v1.joblib
│   ├── credit_default_v2.joblib
│   ├── metrics.json
│   └── sample_request.json
├── scripts/monitor.py
├── src/credit_default_service/
│   ├── app.py
│   ├── config.py
│   ├── features.py
│   ├── inference.py
│   └── train.py
├── tests/
├── Dockerfile
├── Makefile
├── docker-compose.yml
├── params.yaml
├── pyproject.toml
└── requirements.txt
```

## Локальный запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Обучение моделей:

```bash
python -m credit_default_service.train
```

Запуск API:

```bash
python -m credit_default_service
```

Сервис будет доступен по адресу `http://127.0.0.1:5000`.

## Команды Makefile

```bash
make install
make train
make api
make test
make docker-build
make docker-run
make monitor
```

## API

### `GET /`

Возвращает краткую информацию о сервисе и доступных эндпоинтах.

```bash
curl http://127.0.0.1:5000/
```

### `GET /health`

Проверяет готовность сервиса и список загруженных версий модели.

```bash
curl http://127.0.0.1:5000/health
```

Пример ответа:

```json
{
  "available_models": ["v1", "v2"],
  "model_load_error": null,
  "service": "credit-default-predictor",
  "status": "ok"
}
```

### `POST /predict`

Выполняет прогноз для одного клиента. В запросе можно явно указать `model_version`; если версия не указана, сервис распределяет запрос между `v1` и `v2` через стабильное A/B-разделение 50/50.

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d @models/sample_request.json
```

Формат запроса:

```json
{
  "model_version": "v1",
  "features": {
    "ID": 1,
    "LIMIT_BAL": 20000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -2,
    "PAY_6": -2,
    "BILL_AMT1": 3913.0,
    "BILL_AMT2": 3102.0,
    "BILL_AMT3": 689.0,
    "BILL_AMT4": 0.0,
    "BILL_AMT5": 0.0,
    "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0,
    "PAY_AMT2": 689.0,
    "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0,
    "PAY_AMT5": 0.0,
    "PAY_AMT6": 0.0
  }
}
```

Пример ответа:

```json
{
  "ab_group": "control",
  "default_probability": 0.852718,
  "model_version": "v1",
  "prediction": 1,
  "risk_level": "high",
  "threshold": 0.5
}
```

### `POST /batch_predict`

Выполняет прогноз для нескольких клиентов.

```bash
curl -X POST http://127.0.0.1:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"model_version":"v2","items":[{"features":{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,"BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}}]}'
```

## Docker

Docker Hub image: batonhleba/mephi_ml:latest

Сборка и запуск:

```bash
docker build -t batonhleba/mephi_ml:latest .
docker run --rm -p 5000:5000 batonhleba/mephi_ml:latest
```

Запуск через Docker Compose:

```bash
docker compose up --build
```

Публикация образа:

```bash
docker login
docker push batonhleba/mephi_ml:latest
```

## Метрики моделей

Текущие метрики на тестовой выборке сохранены в `models/metrics.json`.

```json
{
  "v1": {
    "accuracy": 0.744,
    "precision": 0.4418,
    "recall": 0.5983,
    "f1": 0.5083,
    "roc_auc": 0.7481
  },
  "v2": {
    "accuracy": 0.7647,
    "precision": 0.4752,
    "recall": 0.6142,
    "f1": 0.5358,
    "roc_auc": 0.7779
  }
}
```

## Тесты

```bash
pytest
```

## Мониторинг

В проекте есть простой вспомогательный скрипт для оценки drift признаков через PSI:

```bash
python scripts/monitor.py
```
