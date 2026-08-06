FROM python:3.11-slim

WORKDIR /srv/embeddings

# git for docs clone at startup
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image at build time (ONNX, ~90MB).
# Download-only via huggingface_hub — do NOT instantiate TextEmbedding here:
# onnxruntime import SIGILLs on non-AVX build hosts (exit 132). The pod runs
# on an AVX node (rook.io/avx=true nodeSelector) where the session loads fine.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='qdrant/all-MiniLM-L6-v2-onnx', cache_dir='/models')"

COPY app.py .

ENV MODEL_CACHE=/models \
    DATA_DIR=/data \
    HF_HUB_OFFLINE=1

EXPOSE 8080
CMD ["python", "app.py"]
