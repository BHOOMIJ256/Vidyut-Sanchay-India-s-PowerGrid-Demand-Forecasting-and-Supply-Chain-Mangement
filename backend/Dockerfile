# 1. Use a lightweight Python version
FROM python:3.11-slim

# 2. Set the working folder inside the container
WORKDIR /app

# 3. Copy requirements first (to use Docker cache efficiently)
COPY requirements.txt .

# 4. Install dependencies (Clean install to avoid bloat)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your backend code into the container
COPY . .

# 6. Expose the port used by FastAPI
EXPOSE 8000

# 7. Start the Server
# We use host 0.0.0.0 so the container can talk to the outside world
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]