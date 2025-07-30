FROM python:3.9-slim-bullseye


COPY --from=ghcr.io/openfaas/of-watchdog:0.10.7 /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /home/app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-handler-url https://download.pytorch.org/whl/cpu

COPY handler.py .

RUN chown -R app:app .

USER app

ENV fprocess="python handler.py"
EXPOSE 8080

CMD ["fwatchdog"]
