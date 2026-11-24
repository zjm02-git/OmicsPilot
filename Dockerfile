FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
COPY . .
EXPOSE 8000
CMD ["uvicorn","web:app","--host","0.0.0.0","--port","8000"]
