FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies
RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean

# Copy files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r deployment/requirements.txt

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "api.ux_agent_api:app", "--host", "0.0.0.0", "--port", "8000"]
