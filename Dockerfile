FROM python:3.12-slim

RUN pip install --no-cache-dir uv
# use the image's Python, never download another one
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Install only the runtime dependencies (no Jupyter etc.) into /app/.venv
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

COPY features.py predict.py model.bin ./

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
