#!/bin/bash

# Script to start a local Qdrant instance using Docker for persistence

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
CONTAINER_NAME="rag_pipeline_qdrant"
QDRANT_IMAGE="qdrant/qdrant:latest" # Use the latest image or pin to a specific version e.g., v1.9.1
# Default storage path (relative to the script location). Can be overridden by an argument.
DEFAULT_STORAGE_PATH="./qdrant_storage"
# Get storage path from the first argument, or use the default
STORAGE_PATH="${1:-$DEFAULT_STORAGE_PATH}"
HOST_PORT_HTTP="6333"
HOST_PORT_GRPC="6334"
CONTAINER_PORT_HTTP="6333"
CONTAINER_PORT_GRPC="6334"

# --- Ensure Docker is running ---
if ! docker info > /dev/null 2>&1; then
  echo "Docker does not seem to be running, please start Docker and try again." >&2
  exit 1
fi

# --- Prepare Storage Directory ---
# Get the absolute path for the storage directory
ABS_STORAGE_PATH="$(cd "$(dirname "$STORAGE_PATH")" && pwd)/$(basename "$STORAGE_PATH")"

echo "Ensuring Qdrant storage directory exists at: ${ABS_STORAGE_PATH}"
mkdir -p "${ABS_STORAGE_PATH}"

# --- Stop and Remove Existing Container (if any) ---
echo "Checking for existing container named '${CONTAINER_NAME}'..."
if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
    echo "Stopping existing container '${CONTAINER_NAME}'..."
    docker stop "${CONTAINER_NAME}"
fi

if [ "$(docker ps -aq -f status=exited -f name=${CONTAINER_NAME})" ]; then
    echo "Removing existing container '${CONTAINER_NAME}'..."
    docker rm "${CONTAINER_NAME}"
fi

# --- Start New Qdrant Container ---
echo "Starting new Qdrant container '${CONTAINER_NAME}'..."
echo "  Image: ${QDRANT_IMAGE}"
echo "  HTTP Port: ${HOST_PORT_HTTP}:${CONTAINER_PORT_HTTP}"
echo "  GRPC Port: ${HOST_PORT_GRPC}:${CONTAINER_PORT_GRPC}"
echo "  Storage Mount: ${ABS_STORAGE_PATH}:/qdrant/storage"

docker run -d --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT_HTTP}:${CONTAINER_PORT_HTTP}" \
  -p "${HOST_PORT_GRPC}:${CONTAINER_PORT_GRPC}" \
  -v "${ABS_STORAGE_PATH}:/qdrant/storage:z" \
  "${QDRANT_IMAGE}"

echo "Qdrant container '${CONTAINER_NAME}' started successfully."
echo "HTTP API available at: http://localhost:${HOST_PORT_HTTP}"
echo "gRPC API available at: localhost:${HOST_PORT_GRPC}" 