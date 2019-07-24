#/bin/sh
docker build -t predict-service .
docker tag predict-service gcr.io/PROJECT_ID/predict-service
gcloud docker -- push gcr.io/PROJECT_ID/predict-service
