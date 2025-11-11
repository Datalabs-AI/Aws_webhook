FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for PyMuPDF (image processing) and PIL
RUN apt-get update && apt-get install -y \
    libfreetype6-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "updated_with_fastapimcp:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
