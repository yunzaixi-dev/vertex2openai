---
title: OpenAI to Gemini Adapter
emoji: 🔄☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# OpenAI to Gemini Adapter

This service provides an OpenAI-compatible API that translates requests to Google's Vertex AI Gemini models, allowing you to use Gemini models with tools expecting an OpenAI interface.

## Features

-   OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/models`).
-   Supports Google Cloud credentials via `GOOGLE_CREDENTIALS_JSON` secret (recommended for Spaces) or local file methods.
-   Supports credential rotation when using local files.
-   Handles streaming and non-streaming responses.
-   Configured for easy deployment on Hugging Face Spaces using Docker (port 7860) or locally via Docker Compose (port 8050).

## Hugging Face Spaces Deployment (Recommended)

This application is ready for deployment on Hugging Face Spaces using Docker.

1.  **Create a new Space:** Go to Hugging Face Spaces and create a new Space, choosing "Docker" as the Space SDK.
2.  **Upload Files:** Upload the `app/` directory, `Dockerfile`, and `app/requirements.txt` to your Space repository. You can do this via the web interface or using Git.
3.  **Configure Secrets:** In your Space settings, navigate to the **Secrets** section and add the following secrets:
    *   `API_KEY`: Your desired API key for authenticating requests to this adapter service. If not set, it defaults to `123456`.
    *   `GOOGLE_CREDENTIALS_JSON`: The **entire content** of your Google Cloud service account JSON key file. Copy and paste the JSON content directly into the secret value field. **This is the required method for providing credentials on Hugging Face.**
4.  **Deployment:** Hugging Face will automatically build and deploy the Docker container. The application will run on port 7860 as defined in the `Dockerfile` and this README's metadata.

Your adapter service will be available at the URL provided by your Hugging Face Space (e.g., `https://your-user-name-your-space-name.hf.space`).

## Local Docker Setup (for Development/Testing)

### Prerequisites

-   Docker and Docker Compose
-   Google Cloud service account credentials with Vertex AI access

### Credential Setup (Local Docker)

1.  Create a `credentials` directory in the project root:
    ```bash
    mkdir -p credentials
    ```
2.  Add your service account JSON files to the `credentials` directory:
    ```bash
    # Example with multiple credential files
    cp /path/to/your/service-account1.json credentials/service-account1.json
    cp /path/to/your/service-account2.json credentials/service-account2.json
    ```
    The service will automatically detect and rotate through all `.json` files in this directory if the `GOOGLE_CREDENTIALS_JSON` environment variable is *not* set.
3.  Alternatively, set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable *in your local environment or `docker-compose.yml`* to the *path* of a single credential file (used as a fallback if the other methods fail).

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
-   `GET /health` - Health check endpoint (includes credential status)

All endpoints require authentication using an API key in the Authorization header.

### Authentication

The service requires an API key for authentication.

To authenticate, include the API key in the `Authorization` header using the `Bearer` token format:

```
Authorization: Bearer YOUR_API_KEY
```

Replace `YOUR_API_KEY` with the key you configured (either via the `API_KEY` secret/environment variable or the default `123456`).

### Example Requests

*(Replace `YOUR_ADAPTER_URL` with your Hugging Face Space URL or `http://localhost:8050` if running locally)*

#### Basic Request

```bash
curl -X POST YOUR_ADAPTER_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gemini-1.5-pro",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7
  }'
```

#### Grounded Search Request

```bash
curl -X POST YOUR_ADAPTER_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "model": "gemini-2.5-pro-exp-03-25-search",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant with access to the latest information."},
      {"role": "user", "content": "What are the latest developments in quantum computing?"}
    ],
    "temperature": 0.2
  }'
```

### Supported Models

The API supports the following Vertex AI Gemini models:

| Model ID                       | Description                                    |
| ------------------------------ | ---------------------------------------------- |
| `gemini-2.5-pro-exp-03-25`     | Gemini 2.5 Pro Experimental (March 25)         |
| `gemini-2.5-pro-exp-03-25-search` | Gemini 2.5 Pro with Google Search grounding |
| `gemini-2.0-flash`             | Gemini 2.0 Flash                             |
| `gemini-2.0-flash-search`      | Gemini 2.0 Flash with Google Search grounding |
| `gemini-2.0-flash-lite`        | Gemini 2.0 Flash Lite                          |
| `gemini-2.0-flash-lite-search` | Gemini 2.0 Flash Lite with Google Search grounding |
| `gemini-2.0-pro-exp-02-05`     | Gemini 2.0 Pro Experimental (February 5)      |
| `gemini-1.5-flash`             | Gemini 1.5 Flash                             |
| `gemini-1.5-flash-8b`          | Gemini 1.5 Flash 8B                          |
| `gemini-1.5-pro`             | Gemini 1.5 Pro                               |
| `gemini-1.0-pro-002`           | Gemini 1.0 Pro                               |
| `gemini-1.0-pro-vision-001`    | Gemini 1.0 Pro Vision                        |
| `gemini-embedding-exp`         | Gemini Embedding Experimental                |

Models with the `-search` suffix enable grounding with Google Search using dynamic retrieval.

### Supported Parameters

The API supports common OpenAI-compatible parameters, mapping them to Vertex AI where possible:

| OpenAI Parameter    | Vertex AI Parameter | Description                                       |
| ------------------- | --------------------- | ------------------------------------------------- |
| `temperature`       | `temperature`         | Controls randomness (0.0 to 1.0)                  |
| `max_tokens`        | `max_output_tokens`   | Maximum number of tokens to generate              |
| `top_p`             | `top_p`               | Nucleus sampling parameter (0.0 to 1.0)           |
| `top_k`             | `top_k`               | Top-k sampling parameter                          |
| `stop`              | `stop_sequences`      | List of strings that stop generation when encountered |
| `presence_penalty`  | `presence_penalty`    | Penalizes repeated tokens                         |
| `frequency_penalty` | `frequency_penalty`   | Penalizes frequent tokens                         |
| `seed`              | `seed`                | Random seed for deterministic generation          |
| `logprobs`          | `logprobs`            | Number of log probabilities to return           |
| `n`                 | `candidate_count`     | Number of completions to generate               |

## Credential Handling Priority

The application loads Google Cloud credentials in the following order:

1.  **`GOOGLE_CREDENTIALS_JSON` Environment Variable / Secret:** Checks for the JSON *content* directly in this variable (Required for Hugging Face).
2.  **`credentials/` Directory (Local Only):** Looks for `.json` files in the directory specified by `CREDENTIALS_DIR` (Default: `/app/credentials` inside the container). Rotates through found files. Used if `GOOGLE_CREDENTIALS_JSON` is not set.
3.  **`GOOGLE_APPLICATION_CREDENTIALS` Environment Variable (Local Only):** Checks for a *file path* specified by this variable. Used as a fallback if the above methods fail.

## Environment Variables / Secrets

-   `API_KEY`: API key for authentication (Default: `123456`). **Required as Secret on Hugging Face.**
-   `GOOGLE_CREDENTIALS_JSON`: **(Required Secret on Hugging Face)** The full JSON content of your service account key. Takes priority over other methods.
-   `CREDENTIALS_DIR` (Local Only): Directory containing credential files (Default: `/app/credentials` in the container). Used if `GOOGLE_CREDENTIALS_JSON` is not set.
-   `GOOGLE_APPLICATION_CREDENTIALS` (Local Only): Path to a *specific* credential file. Used as a fallback if the above methods fail.
-   `PORT`: Not needed for `CMD` config (uses 7860). Hugging Face provides this automatically, `docker-compose.yml` maps 8050 locally.

## Health Check

You can check the status of the service using the health endpoint:

```bash
curl YOUR_ADAPTER_URL/health -H "Authorization: Bearer YOUR_API_KEY"
```

This returns information about the credential status:

```json
{
  "status": "ok",
  "credentials": {
    "available": 1, // Example: 1 if loaded via JSON secret, or count if loaded from files
    "files": [], // Lists files only if using CREDENTIALS_DIR method
    "current_index": 0
  }
}
```

## License

This project is licensed under the MIT License.