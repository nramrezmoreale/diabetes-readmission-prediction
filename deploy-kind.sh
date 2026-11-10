#!/usr/bin/env bash
# Build the image, create a local Kubernetes cluster with kind and deploy the service.
# Requirements: docker, kind (https://kind.sigs.k8s.io), kubectl
set -euo pipefail

CLUSTER=readmission
IMAGE=diabetes-readmission:v1

docker build -t "$IMAGE" .

if ! kind get clusters | grep -qx "$CLUSTER"; then
  kind create cluster --name "$CLUSTER" --config k8s/kind-config.yaml
fi

# make the local image available inside the cluster (no registry needed)
kind load docker-image "$IMAGE" --name "$CLUSTER"

kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl rollout status deployment/readmission-api --timeout=180s
kubectl get pods,svc -l app=readmission-api

echo
echo "Service available at http://localhost:30080 (docs: http://localhost:30080/docs)"
echo "Test it with: uv run python test_predict.py http://localhost:30080"
