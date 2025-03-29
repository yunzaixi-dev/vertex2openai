# OpenAI to Gemini Adapter

This service provides an OpenAI-compatible API that translates requests to Google's Vertex AI Gemini models.

## Features

- OpenAI-compatible API endpoints
- Support for multiple Google Cloud service account credentials with automatic rotation
- Streaming and non-streaming responses
- Docker containerization for easy deployment

## Setup

### Prerequisites

- Docker and Docker Compose
- Google Cloud service account credentials with Vertex AI access

### Credential Setup

The service supports multiple Google Cloud service account credentials and will rotate between them for API requests. This helps distribute API usage across multiple accounts.

1. Create a `credentials` directory in the project root:
   ```
   mkdir -p credentials
   ```

2. Add your service account JSON files to the credentials directory:
   ```
   # Example with multiple credential files
   cp /path/to/your/service-account1.json credentials/service-account1.json
   cp /path/to/your/service-account2.json credentials/service-account2.json
   cp /path/to/your/service-account3.json credentials/service-account3.json
   ```

   The service will automatically detect all `.json` files in the credentials directory.

3. For backward compatibility, you can still name one of your files `service-account.json`:
   ```
   cp /path/to/your/primary-account.json credentials/service-account.json
   ```

## Running the Service

Start the service using Docker Compose:

```
docker-compose up -d
```

The service will be available at http://localhost:8050.

## API Usage

The service implements OpenAI-compatible endpoints:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Create a chat completion
- `GET /health` - Health check endpoint (includes credential status)

All endpoints require authentication using an API key in the Authorization header.

### Authentication

The service requires an API key for authentication. The default API key is `123456`.

To authenticate, include the API key in the Authorization header using the Bearer token format:

```
Authorization: Bearer 123456
```

You can change the API key by setting the `API_KEY` environment variable in the docker-compose.yml file.

### Example Request

```bash
curl -X POST http://localhost:8050/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 123456" \
  -d '{
    "model": "gemini-1.5-pro",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7
  }'
```

### Supported Parameters

The API supports the following OpenAI-compatible parameters that map to Vertex AI's GenerativeModel:

| OpenAI Parameter | Vertex AI Parameter | Description |
|------------------|---------------------|-------------|
| `temperature` | `temperature` | Controls randomness (0.0 to 1.0) |
| `max_tokens` | `max_output_tokens` | Maximum number of tokens to generate |
| `top_p` | `top_p` | Nucleus sampling parameter (0.0 to 1.0) |
| `top_k` | `top_k` | Top-k sampling parameter |
| `stop` | `stop_sequences` | List of strings that stop generation when encountered |
| `presence_penalty` | `presence_penalty` | Penalizes repeated tokens |
| `frequency_penalty` | `frequency_penalty` | Penalizes frequent tokens |
| `seed` | `seed` | Random seed for deterministic generation |
| `logprobs` | `logprobs` | Number of log probabilities to return |
| `n` | `candidate_count` | Number of completions to generate |

## Credential Rotation

The service automatically rotates through all available credential files for each request. If a credential fails, it will try the next one.

You can check the status of credentials using the health endpoint:

```bash
curl http://localhost:8050/health
```

This will return information about the available credentials:

```json
{
  "status": "ok",
  "credentials": {
    "available": 3,
    "files": ["service-account1.json", "service-account2.json", "service-account3.json"],
    "current_index": 1
  }
}
```

## Environment Variables

- `CREDENTIALS_DIR`: Directory containing credential files (default: `/app/credentials`)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to a specific credential file (used as fallback)
- `API_KEY`: API key for authentication (default: `123456`)

## License

This project is licensed under the MIT License - see the LICENSE file for details.