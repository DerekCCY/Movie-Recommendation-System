#!/bin/bash
# Deploy newly trained model to production
# Usage: ./scripts/deploy_model.sh <model_version>
#   e.g., ./scripts/deploy_model.sh recsys-20251113-1430

set -e

MODEL_VERSION=$1

# 6. Restart service
echo "Restarting service..."

for x in 1 2 3; do
    VERSION_STRING="$x.0"
    CONTAINER_IDS=$(docker ps -a -q --filter "name=-$VERSION_STRING")

    for ID in $CONTAINER_IDS; do
        if docker stop "$ID"; then
            echo "Container $VERSION_STRING stopped"
        else
            echo "Container $VERSION_STRING failed to stop"
            continue
        fi
        if docker rm "$ID"; then
            echo "Container $VERSION_STRING removed"
        else
            echo "Container $VERSION_STRING failed to be removed"
            continue
        fi
    done

    PORT=$((8082 + $x))

    TAG="$DOCKER_USERNAME/movie:$MODEL_VERSION-$VERSION_STRING"

    docker build -t $TAG -f "deployment/Dockerfile" .
    sudo docker run -p $PORT:8082 --name "movie-deployment-$VERSION_STRING" "$DOCKER_USERNAME/movie:$MODEL_VERSION-$VERSION_STRING" &

    # 7. Health check
    echo "Performing health check..."
    sleep 5

    HEALTH_URL="http://localhost:$PORT/health"
    HEALTH_PASSED=false

    for i in $(seq 1 10); do
        if curl -f $HEALTH_URL > /dev/null 2>&1; then
            echo "Health check passed (attempt $i/10)"
            HEALTH_PASSED=true
            break
        fi
        echo "Waiting for service... ($i/10)"
        sleep 2
    done

    if [ "$HEALTH_PASSED" = true ]; then
        echo "============================================================"
        echo "Container $VERSION_STRING Deployment Complete"
        echo "============================================================"
        echo "Model: $MODEL_VERSION"
        echo "Status: Running"
        echo "Health: OK"
    else
        echo "============================================================"
        echo "ERROR: Health check failed"
        echo "============================================================"
        echo "Stopping deployment of new containers..."

        exit 1
    fi
done