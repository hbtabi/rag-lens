# Getting Started

This guide walks a new team through their first deployment on Lumen Cloud.

## Create a workspace

Sign up with your work email and create a workspace. The person who creates it becomes the owner and can
invite teammates as admins, developers or read-only viewers.

## Install the CLI

Install the command-line tool with `pipx install lumen-cli` or download a binary from the releases page.
Run `lumen login` to authenticate through your browser; the token is stored in your system keychain.

## Deploy your first app

Run `lumen init` in your project folder to generate a lumen.toml file, then `lumen deploy`. Builds use
the detected language runtime, and each deployment gets a unique preview URL. Promote a preview to
production with `lumen promote`.

## Environment variables

Secrets are set with `lumen env set KEY=value` and are injected at runtime. They are encrypted and never
shown in build logs.
