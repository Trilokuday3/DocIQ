# Build from the repo root:  docker build -f api/Dockerfile -t dociq .
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
# CPU-only torch wheel: the default PyPI wheel drags in GBs of CUDA libraries the API doesn't use.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[infer,serve]"
COPY api/ api/
ENV MODEL_ID=your-hf-username/dociq-donut-cord
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
