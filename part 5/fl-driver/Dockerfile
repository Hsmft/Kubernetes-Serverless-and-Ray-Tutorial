FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy driver code
COPY driver.py .

# Set environment
ENV PYTHONUNBUFFERED=1

CMD ["python", "driver.py"]
