FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir \
      torch==2.7.0 --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
      langchain langchain-huggingface transformers bsedata rapidfuzz


COPY src/ /bazarai/src/
WORKDIR /bazarai/src
RUN pip install -r requirements.txt