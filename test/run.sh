locust -f test/locust.py \
  --headless -u 50 -r 50 -i 50 \
  --host=http://0.0.0.0:8080 \
  --html=report.html