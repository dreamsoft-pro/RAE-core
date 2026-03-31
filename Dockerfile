FROM python:3.14-rc-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir pydantic>=2.0.0 setuptools wheel
RUN pip install -e .
CMD ["python3", "-c", "import rae_core; print('RAE-core v1.1.0 DNA Active')"]
