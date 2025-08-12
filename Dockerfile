FROM python:3.12-slim

WORKDIR /app

# Install libgomp1 for OpenMP support (needed by LightGBM)
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash", "scripts/run_all.sh"]
