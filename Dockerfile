FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/

EXPOSE 7860

CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "1200", "--access-logfile", "-", "--error-logfile", "-"]