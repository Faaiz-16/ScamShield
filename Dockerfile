# One image for both the API and the Telegram bot; docker-compose.yml picks the command.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Trained models are not in the repository, so train the baseline here, from the seed
# sample plus every reviewed label file in data/labels/.
# It takes a few seconds and matches the scikit-learn version installed above.
RUN python -m ml.prepare_data --output-dir /tmp/processed \
    && python -m ml.train_baseline \
        --train /tmp/processed/train.csv \
        --output ml/models/baseline.joblib \
        --results-dir /tmp/train-results \
    && rm -rf /tmp/processed /tmp/train-results

RUN useradd --system --uid 10001 --no-create-home scamshield
USER scamshield

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
