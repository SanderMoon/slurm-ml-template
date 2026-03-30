#!/bin/bash
# Build and push the ml-base Docker image.
#
# This builds a shared base image with common ML dependencies. Your team
# runs this once, pushes to a shared registry, and all generated projects
# reference the image via the base_docker_image copier prompt.
#
# Usage:
#   REGISTRY=ghcr.io NAMESPACE=myorg ./build-ml-base.sh
#   REGISTRY=ghcr.io NAMESPACE=myorg BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime ./build-ml-base.sh
#   ./build-ml-base.sh --no-push     # build only, don't push

set -euo pipefail

REGISTRY="${REGISTRY:?Set REGISTRY (e.g. ghcr.io, registry.example.com:5005)}"
NAMESPACE="${NAMESPACE:?Set NAMESPACE (e.g. your username or org)}"
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime}"
TAG="${TAG:-latest}"
PUSH=true

for arg in "$@"; do
    case "$arg" in
        --no-push) PUSH=false ;;
        --tag=*) TAG="${arg#--tag=}" ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

IMAGE="${REGISTRY}/${NAMESPACE}/ml-base:${TAG}"

echo "=== Building ml-base ==="
echo "  Base image: ${BASE_IMAGE}"
echo "  Target:     ${IMAGE}"
docker build --platform linux/amd64 \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -f Dockerfile.ml-base \
    -t ml-base:latest \
    -t "${IMAGE}" .

if [ "$PUSH" = true ]; then
    echo "=== Pushing ${IMAGE} ==="
    docker push "${IMAGE}"
fi

echo ""
echo "Done! ${IMAGE}"
echo ""
echo "Use this as base_docker_image when creating projects:"
echo "  copier copy --trust -d base_docker_image=\"${IMAGE}\" ..."
