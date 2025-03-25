# dePrompt - AI Prompt Engineering Assistant

A tool to help optimize prompts for different AI models by adding structure, precision, model-specific optimizations, and guardrails.

Developed by Purav Bhardwaj

## Features

- Improve prompts for specific AI models
- Context gathering through intelligent questioning
- Detailed explanations of prompt improvements
- Model-specific optimizations
- Validation and quality assessment

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   python app.py
   ```

3. Access the application at `http://127.0.0.1:5002`

## Deploying to Supabase

1. Create a Supabase account and create a new project.

2. Install Supabase CLI:
   ```
   npm install -g supabase
   ```

3. Log in to Supabase:
   ```
   supabase login
   ```

4. Link your project:
   ```
   supabase link --project-ref your-project-id
   ```

5. Deploy to Supabase:
   ```
   supabase functions deploy deprompt
   ```

6. Set your OpenAI API key as an environment variable:
   ```
   supabase secrets set OPENAI_API_KEY=your-openai-api-key
   ```

## Alternative Deployment with Docker

You can also deploy this application using the provided Dockerfile:

1. Build the Docker image:
   ```
   docker build -t deprompt .
   ```

2. Run the container:
   ```
   docker run -p 8080:8080 -e OPENAI_API_KEY=your-api-key deprompt
   ```

3. Access the application at `http://localhost:8080`

## License

Copyright (c) 2025 Purav Bhardwaj. All rights reserved. 