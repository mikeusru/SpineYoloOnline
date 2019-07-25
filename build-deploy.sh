#/bin/sh
docker build -t predict-service .
docker tag predict-service gcr.io/spine-yolo/predict-service
gcloud docker -- push gcr.io/spine-yolo/predict-service
