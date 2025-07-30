# Use official Python base image for backend
FROM python:3.11-slim AS backend

# Set workdir
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY server/ ./server/
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Frontend build stage ---
FROM node:20-slim AS frontend
WORKDIR /frontend
COPY package.json package-lock.json* vite.config.js .
COPY public ./public
COPY src ./src
RUN npm install && npm run build

# --- Final stage ---
FROM python:3.11-slim
WORKDIR /app

# Copy backend from backend stage
COPY --from=backend /app/server ./server
COPY --from=backend /app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy frontend build from frontend stage
COPY --from=frontend /frontend/dist ./frontend_dist
COPY index.html ./index.html

# Expose backend port
EXPOSE 8000

# Start FastAPI backend (adjust if you use another server)
CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]
