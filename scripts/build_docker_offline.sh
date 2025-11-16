#!/usr/bin/env bash
set -euo pipefail

# Build x86_64 (linux/amd64) image and export as a tarball for offline load.

IMAGE_NAME=${IMAGE_NAME:-capacity-compass}
IMAGE_TAG=${IMAGE_TAG:-latest}
PLATFORM=${PLATFORM:-linux/amd64}
OUT_TAR=${OUT_TAR:-capacity_compass_${IMAGE_TAG}_amd64.tar}

echo "[+] Building ${IMAGE_NAME}:${IMAGE_TAG} for ${PLATFORM} (USTC mirror)"
docker buildx build --platform "${PLATFORM}" \
  --build-arg PIP_INDEX_URL="https://mirrors.ustc.edu.cn/pypi/web/simple" \
  --build-arg http_proxy= --build-arg https_proxy= \
  --build-arg HTTP_PROXY= --build-arg HTTPS_PROXY= \
  --build-arg no_proxy= --build-arg NO_PROXY= \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" . --load

echo "[+] Saving image to ${OUT_TAR}"
docker save -o "${OUT_TAR}" "${IMAGE_NAME}:${IMAGE_TAG}"

echo "[✓] Done. Load on server with: docker load -i ${OUT_TAR}"
echo "    Run: docker run -d --name capacity-compass -p 9050:9050 --env-file .env.runtime ${IMAGE_NAME}:${IMAGE_TAG}"
