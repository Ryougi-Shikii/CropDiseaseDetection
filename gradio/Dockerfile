FROM python:3.10-slim

# Hugging Face Spaces default user (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR $HOME/app

# Upgrade pip and install CPU-optimized PyTorch
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining application dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user:user . .

# Hugging Face standard port
EXPOSE 7860

CMD ["python", "app.py"]
