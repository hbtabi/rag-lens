# Security

Security is a shared responsibility: Lumen Cloud secures the platform, and you secure what you deploy.

## Encryption

All data is encrypted at rest with AES-256 using keys managed in a hardware security module. Traffic
between clients and the platform uses TLS 1.3, and internal service-to-service traffic is protected with
mutual TLS. Customer-managed keys (BYOK) are available on the Enterprise plan.

## Single sign-on

SSO supports SAML 2.0 and OpenID Connect providers such as Okta, Microsoft Entra ID and Google
Workspace. Once SSO is enforced, password logins are disabled for every member except the break-glass
owner account.

## Two-factor authentication

Two-factor authentication can be enabled with an authenticator app or a hardware security key
(WebAuthn). Owners can require 2FA for the entire workspace from the Members page.

## Reporting a vulnerability

Please report vulnerabilities to security@lumen.example. We acknowledge reports within one business day
and run a private bug bounty with rewards of up to £5,000 for critical findings.
