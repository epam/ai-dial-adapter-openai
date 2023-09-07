locust -f test/locust/app.py \
  --headless -u 50 -r 50 -i 50 \
  --host=http://0.0.0.0:8080 \
  --html=test/locust/~report.html