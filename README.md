# Credit Default Prediction Service

Учебный production-like проект по внедрению модели машинного обучения для прогнозирования дефолта по кредитной карте в следующем месяце.

Датасет: `Default of Credit Card Clients Dataset` из UCI. Целевая переменная: `default.payment.next.month`.

Docker Hub image: `batonhleba/mephi_ml:latest`

## Что входит в проект

- обучение двух моделей бинарной классификации: `v1` и `v2`;
- feature engineering внутри сохранённого `sklearn Pipeline`;
- сохранение моделей в `models/*.joblib`;
- Flask API с эндпоинтами `GET /`, `GET /health`, `POST /predict`, `POST /batch_predict`;
- стабильное 50/50 A/B-распределение трафика, если версия модели не задана явно;
- `risk_level` в ответе API: `low`, `medium`, `high`;
- JSONL-логирование API-запросов в `logs/api_requests.jsonl`;
- Dockerfile, `docker-compose.yml`, Makefile и GitHub Actions CI;
- простой PSI-monitoring helper в `scripts/monitor.py`;
- документация по архитектуре, MLOps-концептам, бизнес-метрикам и A/B-тесту.

## Структура

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
└── requirements.txt
```

## Локальный запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m credit_default_service.train
python -m credit_default_service
```

То же самое через Makefile:

```bash
make install
make train
make api
```

Сервис будет доступен на `http://127.0.0.1:5000`.

## Docker


## API

### GET `/`

Возвращает краткую информацию о сервисе и endpoints.

```bash
curl http://127.0.0.1:5000/
```

### GET `/health`

Проверяет состояние сервиса и доступность сохранённых моделей.

```bash
curl http://127.0.0.1:5000/health
```

Пример ответа:

```json
{
  "status": "ok",
  "service": "credit-default-predictor",
  "available_models": ["v1", "v2"],
  "model_load_error": null
}
```

### POST `/predict`

Можно передавать признаки в поле `features` и явно указать `model_version`.
Если `model_version` не указан, сервис стабильно распределит клиента между `v1` и `v2` в пропорции 50/50.

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d @models/sample_request.json
```

Формат тела:

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
  "model_version": "v1",
  "ab_group": "control",
  "prediction": 1,
  "default_probability": 0.712345,
  "threshold": 0.5,
  "risk_level": "high"
}
```

### POST `/batch_predict`

Batch endpoint принимает массив объектов или envelope `{"items": [...]}`.

```bash
curl -X POST http://127.0.0.1:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"model_version":"v2","items":[{"features":{"LIMIT_BAL":20000,"SEX":2,"EDUCATION":2,"MARRIAGE":1,"AGE":24,"PAY_0":2,"PAY_2":2,"PAY_3":-1,"PAY_4":-1,"PAY_5":-2,"PAY_6":-2,"BILL_AMT1":3913,"BILL_AMT2":3102,"BILL_AMT3":689,"BILL_AMT4":0,"BILL_AMT5":0,"BILL_AMT6":0,"PAY_AMT1":0,"PAY_AMT2":689,"PAY_AMT3":0,"PAY_AMT4":0,"PAY_AMT5":0,"PAY_AMT6":0}}]}'
```

## Метрики моделей

После обучения метрики сохраняются в `models/metrics.json`.

Текущие метрики:

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

Основные технические метрики:

- `precision`: доля корректных дефолтов среди клиентов, помеченных рискованными;
- `recall`: доля найденных дефолтов среди всех реальных дефолтов;
- `f1`: баланс precision и recall;
- `roc_auc`: качество ранжирования клиентов по риску.

## Тесты и мониторинг

```bash
pytest
python scripts/monitor.py
```

Через Makefile:

```bash
make test
make monitor
```

## Дополнительная документация

- [Архитектура и MLOps](docs/ARCHITECTURE.md)
- [План A/B-теста](docs/AB_TEST_PLAN.md)
