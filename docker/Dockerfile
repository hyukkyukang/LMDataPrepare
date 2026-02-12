FROM python:3.14-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace

# Keep only packages required to build/install Python deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tzdata \
    ca-certificates && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /workspace/requirements.txt

# Copy repository sources.
COPY . /workspace

# Create an unprivileged runtime user.
RUN useradd -ms /bin/bash user && \
    chown -R user:user /workspace

USER user
WORKDIR /workspace

CMD ["bash"]