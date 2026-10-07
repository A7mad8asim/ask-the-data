FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 ASKDATA_HOME=/app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

COPY knowledge ./knowledge
COPY eval ./eval
COPY app.py ./

# The synthetic data is generated at build time (seeded, about 5 seconds)
RUN python -m askdata.generator

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
