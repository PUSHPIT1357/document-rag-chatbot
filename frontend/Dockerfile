# ---- Build stage ----
FROM oven/bun:1 AS build
WORKDIR /app

# Install deps first so this layer caches across source-only changes.
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

COPY . .

# VITE_* vars are baked into the client bundle at build time, so the API
# origin must be known here, not just at container runtime.
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

# vite.config.ts pins the nitro preset to "node-server" so this produces a
# plain Node server under .output/ instead of Cloudflare's default target.
RUN bun run build

# ---- Runtime stage ----
FROM oven/bun:1-slim AS runtime
WORKDIR /app

ENV NODE_ENV=production \
    PORT=3000 \
    HOST=0.0.0.0

COPY --from=build /app/.output ./.output

EXPOSE 3000

CMD ["bun", "run", ".output/server/index.mjs"]
