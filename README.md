---
title: OpenAI to Gemini Adapter
emoji: 🔄☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860 # Port exposed by Dockerfile, used by Hugging Face Spaces
---

# OpenAI to Gemini Adapter

This service provides an OpenAI-compatible API that translates requests to Google's Vertex AI Gemini models, allowing you to use Gemini models with tools expecting an OpenAI interface. The codebase has been refactored for modularity and improved maintainability.

## Features

-   OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/models`).
-   Modular codebase located within the `app/` directory.
-   Centralized environment variable management in `app/config.py`.
-   Supports Google Cloud credentials via:
    -   `GOOGLE_CREDENTIALS_JSON` environment variable (containing the JSON key content).
    -   Service account JSON files placed in a specified directory (defaults to `credentials/` in the project root, mapped to `/app/credentials` in the container).
-   Supports credential rotation when using multiple local credential files.
-   Handles streaming and non-streaming responses.
-   Configured for easy deployment on Hugging Face Spaces using Docker (port 7860) or locally via Docker Compose (port 8050).
-   Support for Vertex AI Express Mode via `VERTEX_EXPRESS_API_KEY` environment variable.

## Hugging Face Spaces Deployment (Recommended)

This application is ready for deployment on Hugging Face Spaces using Docker.

1.  **Create a new Space:** Go to Hugging Face Spaces and create a new Space, choosing "Docker" as the Space SDK.
2.  **Upload Files:** Add all project files (including the `app/` directory, `.gitignore`, `Dockerfile`, `docker-compose.yml`, and `requirements.txt`) to your Space repository. You can do this via the web interface or using Git.
3.  **Configure Secrets:** In your Space settings, navigate to the **Secrets** section and add the following:
    *   `API_KEY`: Your desired API key for authenticating requests to this adapter service. (Default: `123456` if not set, as per `app/config.py`).
    *   `GOOGLE_CREDENTIALS_JSON`: The **entire content** of your Google Cloud service account JSON key file. This is the primary method for providing credentials on Hugging Face.
    *   `VERTEX_EXPRESS_API_KEY` (Optional): If you have a Vertex AI Express API key and want to use eligible models in Express Mode.
    *   Other environment variables (see "Environment Variables" section below) can also be set as secrets if you need to override defaults (e.g., `FAKE_STREAMING`).
4.  **Deployment:** Hugging Face will automatically build and deploy the Docker container. The application will run on port 7860.

Your adapter service will be available at the URL provided by your Hugging Face Space.

## Local Docker Setup (for Development/Testing)

### Prerequisites

-   Docker and Docker Compose
-   Google Cloud service account credentials with Vertex AI access (if not using Vertex Express exclusively).

### Credential Setup (Local Docker)

The application uses `app/config.py` to manage environment variables. You can set these in a `.env` file at the project root (which is ignored by git) or directly in your `docker-compose.yml` for local development.

1.  **Method 1: JSON Content via Environment Variable (Recommended for consistency with Spaces)**
    *   Set the `GOOGLE_CREDENTIALS_JSON` environment variable to the full JSON content of your service account key.
2.  **Method 2: Credential Files in a Directory**
    *   If `GOOGLE_CREDENTIALS_JSON` is *not* set, the adapter will look for service account JSON files in the directory specified by the `CREDENTIALS_DIR` environment variable.
    *   The default `CREDENTIALS_DIR` is `/app/credentials` inside the container.
    *   Create a `credentials` directory in your project root: `mkdir -p credentials`
    *   Place your service account JSON key files (e.g., `my-project-creds.json`) into this `credentials/` directory. The `docker-compose.yml` mounts this local directory to `/app/credentials` in the container.
    *   The service will automatically detect and rotate through all `.json` files in this directory.

### Environment Variables for Local Docker (`.env` file or `docker-compose.yml`)

Create a `.env` file in the project root or modify your `docker-compose.override.yml` (if you use one) or `docker-compose.yml` to set these:

```env
API_KEY="your_secure_api_key_here" # Replace with your actual key or leave for default
# GOOGLE_CREDENTIALS_JSON='{"type": "service_account", ...}' # Option 1: Paste JSON content
# CREDENTIALS_DIR="/app/credentials" # Option 2: (Default path if GOOGLE_CREDENTIALS_JSON is not set)
# VERTEX_EXPRESS_API_KEY="your_vertex_express_key" # Optional
# FAKE_STREAMING="false" # Optional, for debugging
# FAKE_STREAMING_INTERVAL="1.0" # Optional, for debugging
```

### Running Locally

Start the service using Docker Compose:

```bash
docker-compose up -d
```
The service will be available at `http://localhost:8050` (as defined in `docker-compose.yml`).

## API Usage

The service implements OpenAI-compatible endpoints:

-   `GET /v1/models` - List available models
-   `POST /v1/chat/completions` - Create a chat completion
-   `GET /` - Basic status endpoint

All API endpoints require authentication using an API key in the Authorization header.

### Authentication

Include the API key in the `Authorization` header using the `Bearer` token format:
`Authorization: Bearer YOUR_API_KEY`
Replace `YOUR_API_KEY` with the key configured via the `API_KEY` environment variable (or the default).

### Example Requests

*(Replace `YOUR_ADAPTER_URL` with your Hugging Face Space URL or `http://localhost:8050` if running locally)*

#### Basic Request
```bash
curl -X POST YOUR_ADAPTER_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gemini-1.5-pro", # Or any other supported model
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7
  }'
```

### Supported Models & Parameters
(Refer to the `list_models` endpoint output and original documentation for the most up-to-date list of supported models and parameters. The adapter aims to map common OpenAI parameters to their Vertex AI equivalents.)

## Credential Handling Priority

The application (via `app/config.py` and helper modules) prioritizes credentials as follows:

1.  **Vertex AI Express Mode (`VERTEX_EXPRESS_API_KEY` env var):** If this key is set and the requested model is eligible for Express Mode, this will be used.
2.  **Service Account Credentials (Rotated):** If Express Mode is not used/applicable:
    *   **`GOOGLE_CREDENTIALS_JSON` Environment Variable:** If set, its JSON content is parsed. Multiple JSON objects (comma-separated) or a single JSON object are supported. These are loaded into the `CredentialManager`.
    *   **Files in `CREDENTIALS_DIR`:** The `CredentialManager` scans the directory specified by `CREDENTIALS_DIR` (default is `credentials/` mapped to `/app/credentials` in Docker) for `.json` Mkey files.
    *   The `CredentialManager` then rotates through all successfully loaded service account credentials (from `GOOGLE_CREDENTIALS_JSON` and files in `CREDENTIALS_DIR`) for each request.

## Key Environment Variables

These are sourced by `app/config.py`:

-   `API_KEY`: API key for authenticating to this adapter service. (Default: `123456`)
-   `GOOGLE_CREDENTIALS_JSON`: (Takes priority for SA creds) Full JSON content of your service account key(s).
-   `CREDENTIALS_DIR`: Directory for service account JSON files if `GOOGLE_CREDENTIALS_JSON` is not set. (Default: `/app/credentials` within container context)
-   `VERTEX_EXPRESS_API_KEY`: Optional API key for using Vertex AI Express Mode with compatible models.
-   `FAKE_STREAMING`: Set to `"true"` to enable simulated streaming for non-streaming models (for testing). (Default: `"false"`)
-   `FAKE_STREAMING_INTERVAL`: Interval in seconds for sending keep-alive messages during fake streaming. (Default: `1.0`)

## License

This project is licensed under the MIT License.
