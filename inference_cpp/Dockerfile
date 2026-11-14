# Build stage: compile the server
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake make git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j --target reco_server

# Runtime stage: server binary + embeddings
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=build /src/build/reco_server /app/reco_server

# Embeddings are published as a GitHub Release asset (too large for git)
ARG ARTIFACTS_URL=https://github.com/linhtruong309/shop-recommender/releases/download/v1.0.0/artifacts.tar.gz
RUN mkdir -p /app/artifacts \
    && curl -fsSL "$ARTIFACTS_URL" | tar -xz -C /app/artifacts

ENV PORT=8080 RECO_THREADS=1
EXPOSE 8080
CMD ["/app/reco_server", "/app/artifacts"]
