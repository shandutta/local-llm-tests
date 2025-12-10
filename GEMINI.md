# Gemini Code Assistant Context

This document provides context for the Gemini Code Assistant to understand the `local-llm-tests` project.

## Project Overview

This project is a comprehensive system for managing and interacting with local Large Language Models (LLMs) on a high-end workstation with a powerful NVIDIA GPU. It provides a command-line interface (CLI), a RESTful API, and a web-based UI for a seamless user experience.

The core technologies used in this project are:

*   **`llama.cpp`**: A C/C++ library for running LLMs on local hardware.
*   **Docker**: For containerizing the `llama.cpp` environment, ensuring portability and reproducibility.
*   **Python with FastAPI**: A high-performance web framework used to create a RESTful API that acts as a control plane for managing the LLM lifecycle.
*   **Next.js with React and TypeScript**: A popular web framework used to build a modern and interactive web UI for controlling the LLMs.
*   **Shell Scripts**: For various automation and orchestration tasks.

## Building and Running

The project can be run in several ways, depending on the user's needs.

### Using the Web UI (Recommended)

This is the easiest way to get started. The web UI provides a user-friendly interface for managing and interacting with the LLMs.

1.  **Start the application stack:**

    ```bash
    ./bin/local-llm-launcher.sh start
    ```

2.  **Access the web UI:**

    Open your web browser and navigate to `http://localhost:3000`.

### Using the Command-Line Interface (CLI)

The CLI provides a quick way to manage the LLM lifecycle from the terminal.

*   **List available models:**

    ```bash
    ./bin/local-llm list
    ```

*   **Start a model:**

    ```bash
    ./bin/local-llm start <model_name>
    ```

*   **Stop the running model:**

    ```bash
    ./bin/local-llm stop
    ```

*   **Check the status:**

    ```bash
    ./bin/local-llm status
    ```

### Running the FastAPI Server Directly

This is useful for development and debugging purposes.

1.  **Install the required Python packages:**

    ```bash
    pip install -r server/requirements.txt
    ```

2.  **Start the server:**

    ```bash
    uvicorn server.main:app --reload --port 8008
    ```

### Running the Next.js Frontend Directly

This is useful for frontend development.

1.  **Install the required Node.js packages:**

    ```bash
    cd web
    npm install
    ```

2.  **Start the development server:**

    ```bash
    npm run dev
    ```

## Development Conventions

*   **Configuration:** The core configuration is managed in the `config/models.yaml` file. This file defines the available models, their paths, ports, and `llama.cpp` arguments.
*   **API-Driven:** The web UI is driven by the FastAPI backend. All actions in the UI are mapped to API calls.
*   **Modularity:** The project is well-structured, with separate directories for the server, web UI, configuration, and scripts.
*   **Extensibility:** The project is designed to be extensible. New models can be added by simply updating the `config/models.yaml` file. The web UI even provides a feature to download and register models from Hugging Face.
