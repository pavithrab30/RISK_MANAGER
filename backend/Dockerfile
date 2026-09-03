FROM python:3.11-slim

# libgl1/libglib2.0-0: required by opencv-python (a Docling dependency) for
# image ops even in headless/CPU-only use.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only torch build explicitly, before the rest of the deps -
# otherwise pip resolves the default (CUDA-enabled) wheel, which is several
# GB and pointless for a project that deliberately targets CPU-only
# inference (see ADR.md "Models and providers").
RUN pip install --no-cache-dir "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir -e .

# Torch's compile path needs an MSVC-equivalent toolchain we don't install
# here; eager mode is plenty fast for this project's page counts (see
# app/data/parsing/docling_parser.py for the full rationale).
ENV TORCHDYNAMO_DISABLE=1
ENV TORCH_COMPILE_DISABLE=1
ENV STORAGE_DIR=/data/storage
ENV CHROMA_DIR=/data/storage/chroma
ENV SQLITE_PATH=/data/storage/docintel.sqlite3
ENV UPLOAD_DIR=/data/storage/uploads

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
