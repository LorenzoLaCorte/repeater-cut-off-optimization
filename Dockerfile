FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x ./src/scripts/reproduce_results.sh
RUN mkdir -p /app/results

CMD ["bash", "./src/scripts/reproduce_results.sh"]