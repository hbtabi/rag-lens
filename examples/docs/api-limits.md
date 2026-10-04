# API Rate Limits

The public REST API is rate limited per API token using a token-bucket algorithm.

## Default limits

Starter tokens may send 60 requests per minute with bursts of up to 20. Team tokens may send 600
requests per minute, and Enterprise limits are configured per contract. Uploads larger than 100 MB
must use the multipart upload endpoint.

## Handling 429 responses

When a limit is exceeded the API returns HTTP 429 Too Many Requests with a Retry-After header that gives
the number of seconds to wait. Clients should back off exponentially with jitter rather than retrying
immediately. The official SDKs retry up to three times automatically.

## Response headers

Every response includes X-RateLimit-Limit, X-RateLimit-Remaining and X-RateLimit-Reset headers so you
can track how much of your quota is left in the current window.
