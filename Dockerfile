#Base image 
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy project files 
COPY models/ ./models/
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
