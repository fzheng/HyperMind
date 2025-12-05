# Render Cloud Deployment

This guide covers deploying SigmaPilot to [Render](https://render.com) using the Blueprint specification.

## Prerequisites

- A Render account
- Repository connected to Render (GitHub)
- The `render.yaml` blueprint file in your repository root

## Architecture on Render

```
┌─────────────────────────────────────────────────────────────────┐
│                         Render Cloud                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│   │ hl-scout │    │hl-stream │    │ hl-sage  │    │hl-decide │  │
│   │  :8080   │    │  :8080   │    │  :8080   │    │  :8080   │  │
│   └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘  │
│        │               │               │               │        │
│        └───────────────┼───────────────┼───────────────┘        │
│                        │               │                        │
│                   ┌────┴────┐     ┌────┴────┐                   │
│                   │  NATS   │     │hlbot-db │                   │
│                   │  :4222  │     │(Postgres)│                   │
│                   └─────────┘     └──────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Type | Port | Health Check |
|---------|------|------|--------------|
| nats | Web (Docker) | 4222 | - |
| hl-scout | Web (Docker) | 8080 | `/healthz` |
| hl-stream | Web (Docker) | 8080 | `/healthz` |
| hl-sage | Web (Docker) | 8080 | `/healthz` |
| hl-decide | Web (Docker) | 8080 | `/healthz` |
| hlbot-db | PostgreSQL | 5432 | - |

## Deployment Steps

### 1. Create Blueprint

1. Go to Render Dashboard → **Blueprints**
2. Click **New Blueprint Instance**
3. Select your repository and branch (e.g., `main` or `deploy-test`)
4. Render will detect `render.yaml` automatically

### 2. Configure Secrets

Before deployment completes, set the `OWNER_TOKEN` secret for each service:

1. Go to each service → **Environment**
2. Add `OWNER_TOKEN` with a secure value
3. This token is required for admin endpoints (refresh, seed, etc.)

### 3. Initial Data Population

After all services are healthy, the Alpha Pool will auto-populate on first startup if the database is empty.

To manually refresh:
```bash
curl -X POST "https://hl-sage-xxx.onrender.com/alpha-pool/refresh?limit=50" \
  -H "x-owner-key: YOUR_OWNER_TOKEN"
```

## Environment Variables

### Automatic (Set by Render)

| Variable | Source | Description |
|----------|--------|-------------|
| `DATABASE_URL` | hlbot-db | PostgreSQL connection string |
| `NATS_HOST` | nats service | NATS hostname |
| `SCOUT_HOST` | hl-scout service | Scout hostname |
| `SAGE_HOST` | hl-sage service | Sage hostname |

### Manual (Set in Dashboard)

| Variable | Required | Description |
|----------|----------|-------------|
| `OWNER_TOKEN` | Yes | Admin API authentication token |

### Service-Specific URLs

The `render.yaml` configures inter-service communication:

```yaml
# hl-stream connects to other services
SCOUT_URL: http://$(SCOUT_HOST):8080
SAGE_URL: http://$(SAGE_HOST):8080
NATS_URL: nats://$(NATS_HOST):4222
```

## Database Migrations

Migrations run automatically on hl-scout startup via `runMigrations()`. The schema is tracked in the `schema_migrations` table.

For Python services (hl-sage, hl-decide), an entrypoint script waits for migrations to complete:

```bash
# docker/entrypoint-python.sh
# Waits for position_signals table (migration 014) before starting
```

## Monitoring

### Health Checks

All services expose `/healthz`:
```bash
curl https://hl-stream-xxx.onrender.com/healthz
```

### Logs

View logs in Render Dashboard → Service → **Logs**

Key log patterns to watch:
- `[hl-scout] Migrations complete` - Database ready
- `[hl-sage] Alpha Pool auto-refreshed with N traders` - Pool populated
- `[entrypoint] database schema is ready` - Python services ready

### Metrics

Prometheus metrics available at `/metrics` on each service.

## Troubleshooting

### Services failing to start

**Symptom**: Python services (hl-sage, hl-decide) restart repeatedly

**Cause**: Waiting for database migrations from hl-scout

**Fix**: Check hl-scout logs for migration completion. Services will auto-recover.

### Alpha Pool empty

**Symptom**: Dashboard shows no traders in Alpha Pool

**Possible causes**:
1. Auto-refresh hasn't run yet (wait 5 seconds after startup)
2. Hyperliquid API rate limited during refresh
3. Quality filters too strict (no traders pass 7 gates)

**Debug**:
```bash
# Check hl-sage logs for refresh status
# Look for: "Alpha Pool auto-refreshed with N traders"

# Manual refresh
curl -X POST "https://hl-sage-xxx.onrender.com/alpha-pool/refresh?limit=50" \
  -H "x-owner-key: YOUR_TOKEN"
```

### Service communication errors

**Symptom**: `proxy_failed` errors in hl-stream logs

**Check**:
1. Verify all services are healthy in Render Dashboard
2. Check `SAGE_URL` and `SCOUT_URL` environment variables
3. Ensure services are in the same region (Oregon)

### CRLF line ending errors

**Symptom**: `exec /entrypoint.sh: no such file or directory`

**Cause**: Windows CRLF line endings in shell scripts

**Fix**: The `.gitattributes` file forces LF endings:
```
*.sh text eol=lf
docker/entrypoint*.sh text eol=lf
```

If issue persists, manually convert:
```bash
sed -i 's/\r$//' docker/entrypoint-python.sh
```

## Updating Deployment

### Code Changes

Push to your deployment branch. Render auto-deploys on push.

### Force Rebuild

1. Go to service in Render Dashboard
2. Click **Manual Deploy** → **Clear build cache & deploy**

### Database Reset

To reset the database (development only):
1. Go to hlbot-db in Render Dashboard
2. Delete the database
3. Recreate from Blueprint
4. Redeploy all services

## Cost Optimization

### Starter Plan Limits

- 512 MB RAM per service
- Shared CPU
- Spins down after 15 minutes of inactivity (free tier)

### Recommendations

1. Use Starter plan for testing
2. Upgrade to Standard for production (no spin-down)
3. Consider combining services if hitting limits

## Security Notes

1. **OWNER_TOKEN**: Set unique, strong tokens per environment
2. **DATABASE_URL**: Never expose; use Render's secret management
3. **Public endpoints**: Dashboard is public; admin endpoints require token
4. **HTTPS**: Render provides automatic SSL for all services

## Related Documentation

- [README.md](../README.md) - Project overview
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) - Development roadmap
- [API.md](API.md) - API reference
