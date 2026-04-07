FROM python:3.10-slim

WORKDIR /code

COPY pyproject.toml .
RUN pip install --no-cache-dir fastapi uvicorn requests pydantic openenv-core openai

COPY . .

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
