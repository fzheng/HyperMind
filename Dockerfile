# Multi-stage Dockerfile for hlbot (Node.js + TypeScript)

FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
# Install all deps (incl. dev) for building TypeScript
RUN npm ci

FROM deps AS builder
WORKDIR /app
COPY . .
# Build TS -> JS and prune dev deps for runtime
RUN npm run build && npm prune --omit=dev

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

# Copy production artifacts
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./
COPY scripts ./scripts
# Migrations moved under scripts/migrations; expose env for runner
ENV MIGRATIONS_DIR=/app/scripts/migrations
# (Legacy root migrations directory removed)
COPY docker/entrypoint.sh ./docker/entrypoint.sh

RUN chmod +x ./docker/entrypoint.sh

USER node
EXPOSE 3000
ENTRYPOINT ["/bin/sh", "/app/docker/entrypoint.sh"]
CMD ["node", "dist/server.js"]
