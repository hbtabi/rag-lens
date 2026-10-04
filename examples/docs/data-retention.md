# Data Retention and Privacy

Lumen Cloud processes customer data as a processor under UK GDPR and EU GDPR.

## Log retention

Application logs are retained for 7 days on Starter, 30 days on Team and up to 400 days on Enterprise.
Logs can be streamed to external tools such as Datadog or an S3-compatible bucket.

## Backups

Managed databases are backed up every 24 hours and backups are kept for 14 days. Point-in-time recovery
to any second within the last 7 days is available on Team and Enterprise.

## Deleting your data

When a workspace is deleted, its data is soft-deleted for 30 days so it can be restored, and then
permanently erased from primary storage and backups within a further 35 days.

## Data residency

Workspaces can be pinned to the London (eu-west-2) or Frankfurt (eu-central-1) regions. Data at rest
never leaves the selected region.
