prefect config set PREFECT_SERVER_API_HOST="0.0.0.0"
prefect config set PREFECT_API_URL="http://${EXTERNAL_HOST_NAME}:4200/api"
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://${POSTGRES_USERNAME}:${POSTGRES_PASSWORD}@postgres:5432/prefect"
prefect server start