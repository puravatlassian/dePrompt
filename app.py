from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import time  # For simulating delay during testing

"""
dePrompt - AI Prompt Engineering Assistant
Developed by Purav Bhardwaj

A tool to help optimize prompts for different AI models by adding
structure, precision, model-specific optimizations, and guardrails.

Copyright (c) 2025 Purav Bhardwaj
"""

# Load environment variables
load_dotenv()

# Create Flask application
app = Flask(__name__)

# Get API key and verify it exists
api_key = os.getenv('API_KEY')
if not api_key:
    raise ValueError("API_KEY environment variable is not set")

# Initialize the custom OpenAI client
client = OpenAI(
    api_key=api_key
)

# Model-specific guidance
model_guidance = {
    # Model guidance dictionary 
    # capabilities, optimal uses, context windows, reliability scores,
    # and special considerations for each model
    
    "gpt-4o": {
        "capabilities": ["Multimodal", "advanced reasoning", "200+ languages", "real-time knowledge"],
        "optimal_uses": ["Complex tasks", "image understanding", "multilingual content", "code generation"],
        "context_window": 128000,
        "reliability": 0.96,
        "considerations": """
- Superior performance in most non-English languages
- Strong at reasoning through complex problems visually
- Performs best with clear, focused instructions
- More cost-effective than o1 models for general tasks"""
    },
    "gpt-4o-mini": {
        "capabilities": ["Multimodal", "speed", "cost-effectiveness", "multilingual"],
        "optimal_uses": ["Routine tasks", "image understanding", "high-volume applications"],
        "context_window": 128000,
        "reliability": 0.90,
        "considerations": """
- Excellent cost-to-performance ratio for everyday tasks
- Comparable in many tasks to older GPT-4 versions
- Strong at code generation and factual answers
- Best for applications requiring frequent API calls"""
    },
    "gpt-4.5": {
        "capabilities": ["Advanced reasoning", "creative generation", "factual grounding"],
        "optimal_uses": ["Creative writing", "advanced problem-solving", "strategic analysis"],
        "context_window": 128000,
        "reliability": 0.97,
        "considerations": """
- Represents OpenAI's latest model improvements
- Stronger than GPT-4o in creative and subjective tasks
- Enhanced ability to follow complex instructions precisely
- Improved factual accuracy with reduced hallucinations"""
    },
    "o3-mini": {
        "capabilities": ["Specialized reasoning", "step-by-step problem-solving", "text-only processing"],
        "optimal_uses": ["Mathematics", "coding challenges", "logical reasoning tasks"],
        "context_window": 200000,
        "reliability": 0.99,
        "considerations": """
- Takes longer to respond but provides more methodical answers
- Excels at tasks requiring formal reasoning and precision
- Often works better with explicit reasoning instructions
- More factually accurate than general-purpose models"""
    },
    "o1": {
        "capabilities": ["Expert reasoning", "multimodal understanding", "precise problem-solving"],
        "optimal_uses": ["Complex technical work", "mathematical proofs", "detailed analysis"],
        "context_window": 200000,
        "reliability": 0.98,
        "considerations": """
- Designed for maximum reasoning capability, not speed
- Often benefits from being asked to solve step by step
- Provides detailed explanations of its reasoning process
- Consider the cost tradeoff for simpler tasks"""
    },
    "claude-3.5-sonnet": {
        "capabilities": ["Balanced performance", "long context", "coding excellence", "tool use"],
        "optimal_uses": ["Software development", "research synthesis", "document analysis"],
        "context_window": 200000,
        "reliability": 0.97,
        "considerations": """
- Strong technical accuracy while maintaining approachable tone
- Excellent at understanding and working within guidelines
- Native computer/tool use capabilities for complex tasks
- Maintains context awareness across very lengthy exchanges"""
    },
    "claude-3.7-sonnet": {
        "capabilities": ["Advanced reasoning", "nuanced understanding", "extended thinking mode"],
        "optimal_uses": ["Complex reasoning tasks", "sensitive content moderation", "thorough analysis"],
        "context_window": 200000,
        "reliability": 0.98,
        "considerations": """
- Latest model with superior reasoning capabilities
- Includes an "extended thinking" mode for complex problems
- More factually accurate than previous Claude models
- Excels at carefully weighing evidence and uncertainties"""
    },
    "claude-3-haiku": {
        "capabilities": ["Speed", "cost-effectiveness", "general-purpose"],
        "optimal_uses": ["Chat applications", "content moderation", "summarization"],
        "context_window": 200000,
        "reliability": 0.92,
        "considerations": """
- Fastest Claude model with good balance of quality and speed
- Consider for high-volume, time-sensitive applications
- Strong at following specific tonal and formatting guidance
- May struggle with complex reasoning tasks"""
    },
    "gemini-1.5-pro": {
        "capabilities": ["Long-context reasoning", "multimodal", "multilingual"],
        "optimal_uses": ["Video analysis", "large document processing", "complex research"],
        "context_window": 2000000,
        "reliability": 0.94,
        "considerations": """
- Industry-leading 2M token context window
- Excellent for tasks requiring integration of many documents
- Can process video, audio, and images natively
- Consider using more structured prompts for best results"""
    },
    "gemini-2.0-pro": {
        "capabilities": ["Advanced reasoning", "knowledge-intensive tasks", "coding"],
        "optimal_uses": ["Software development", "complex problem-solving", "technical analysis"],
        "context_window": 2000000,
        "reliability": 0.96,
        "considerations": """
- Google's most advanced model to date
- Exceptional coding capabilities with strong reasoning
- Can use tools like Search and code execution
- Performs best with clear, structured instructions"""
    },
    "gemini-2.0-flash": {
        "capabilities": ["Fast responses", "good reasoning", "multimodal"],
        "optimal_uses": ["Real-time applications", "interactive experiences", "general tasks"],
        "context_window": 1000000,
        "reliability": 0.93,
        "considerations": """
- Optimized for latency-sensitive applications
- Twice as fast as Gemini 1.5 Pro with comparable quality
- Strong balance between performance and efficiency
- Consider for user-facing applications requiring speed"""
    },
    "mistral-large-2": {
        "capabilities": ["Balanced performance", "multilingual", "instruction following"],
        "optimal_uses": ["Enterprise applications", "customer service", "content generation"],
        "context_window": 32000,
        "reliability": 0.95,
        "considerations": """
- Mistral AI's latest enterprise-grade model
- Strong at following precise instructions and constraints
- Excels at maintaining consistent tone and voice
- Good balance of reasoning and creative capabilities"""
    },
    "llama-3.1-405b": {
        "capabilities": ["Open weights", "strong reasoning", "multilingual"],
        "optimal_uses": ["Enterprise deployments", "customized applications", "research"],
        "context_window": 128000,
        "reliability": 0.94,
        "considerations": """
- Meta's largest and most capable open model
- Comparable to closed-source models in many benchmarks
- Can be fine-tuned for specific use cases
- Requires structured prompting for best results"""
    },
    "llama-3.2-1b": {
        "capabilities": ["Efficiency", "compact size", "on-device deployment"],
        "optimal_uses": ["Edge devices", "mobile applications", "embedded systems"],
        "context_window": 128000,
        "reliability": 0.81,
        "considerations": """
- Extremely efficient for size-constrained deployments
- Can run on devices with limited resources
- Best for simpler, well-defined tasks
- Performs better with explicit instruction formats"""
    }
}

# Define the base HTML template with AJAX form submission and dynamic loader
BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>dePrompt - AI Prompt Engineering Assistant</title>
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-E3QSCYVVRG"></script>
    <script>
        window.dataLayer = window.dataLayer || [];
        function gtag(){dataLayer.push(arguments);}
        gtag('js', new Date());
        gtag('config', 'G-E3QSCYVVRG');
    </script>
    <link rel="stylesheet" href="https://unpkg.com/@atlaskit/css-reset@6.0.1/dist/bundle.css" />
    <link rel="stylesheet" href="https://unpkg.com/@atlaskit/tokens@0.13.0/css/tokens.css" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;600&display=swap" rel="stylesheet">
    <style>
        :root {
            color-scheme: dark;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideIn {
            from { transform: translateX(-20px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        body {
            background-color: var(--ds-surface-raised, #1D2125);
            color: var(--ds-text, #C7D1DB);
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0;
            min-height: 100vh;
            line-height: 1.3;
        }

        .navbar {
            background-color: var(--ds-surface-overlay, #22272B);
            border-bottom: 1px solid var(--ds-border, #404040);
            padding: 16px 24px;
            animation: slideIn 0.5s ease-out;
        }

        .nav-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .nav-brand {
            color: var(--ds-text-selected, #DEEBFF);
            font-size: 20px;
            font-weight: 500;
            user-select: none;
        }

        .nav-brand span {
            color: var(--ds-text-subtle, #9FADBC);
            font-weight: 300;
        }

        .nav-links {
            display: flex;
            gap: 24px;
        }

        .nav-link {
            color: var(--ds-text-subtle, #9FADBC);
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            padding: 8px 12px;
            border-radius: 3px;
            transition: all 0.2s ease;
        }

        .nav-link:hover {
            color: var(--ds-text-selected, #DEEBFF);
            background-color: var(--ds-surface-selected, #1D2125);
        }

        .nav-link.active {
            color: var(--ds-text-selected, #DEEBFF);
            background-color: var(--ds-surface-selected, #1D2125);
        }

        .page-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 32px;
        }

        .header {
            margin-bottom: 32px;
            text-align: center;
        }

        h1 {
            color: var(--ds-text-selected, #DEEBFF);
            font-size: 24px;
            font-weight: 500;
            margin-bottom: 4px;
        }

        .brand-prefix {
            color: var(--ds-text-subtle, #9FADBC);
            font-weight: 300;
            margin-right: 0;
        }

        .subtitle {
            color: var(--ds-text-inverse, #FFFFFF);
            font-size: 12px;
            font-weight: 500;
            background-color: var(--ds-background-brand-bold, #0052CC);
            padding: 4px 8px;
            border-radius: 3px;
            margin-top: 8px;
            display: inline-block;
        }

        .form-field {
            margin-bottom: 24px;
        }

        .form-field:last-child {
            margin-bottom: 0;
        }

        label {
            display: block;
            color: var(--ds-text-subtle, #9FADBC);
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 4px;
            text-transform: uppercase;
        }

        .field-description {
            color: var(--ds-text-subtlest, #8C9BAB);
            font-size: 12px;
            margin-bottom: 8px;
        }

        select, textarea, .input-container, .user-input {
            width: 100%;
            box-sizing: border-box;
            padding: 8px 12px;
            border: 2px solid var(--ds-border-input, #404040);
            border-radius: 3px;
            background-color: var(--ds-surface-sunken, #161A1D);
            color: var(--ds-text, #C7D1DB);
            font-size: 14px;
            margin-bottom: 16px;
            transition: all 0.2s ease;
        }

        select:hover, textarea:hover {
            background-color: var(--ds-surface-hovered, #1D2125);
        }

        select:focus, textarea:focus {
            outline: none;
            border-color: var(--ds-border-focused, #4C9AFF);
            background-color: var(--ds-surface-selected, #1D2125);
        }

        button {
            width: 100%;
            padding: 8px 16px;
            background-color: var(--ds-background-brand-bold, #0052CC);
            color: var(--ds-text-inverse, #FFFFFF);
            border: none;
            border-radius: 3px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        button:hover {
            background-color: var(--ds-background-brand-bold-hovered, #0065FF);
            transform: translateY(-1px);
        }

        button:active {
            background-color: var(--ds-background-brand-bold-pressed, #0747A6);
            transform: translateY(0);
        }

        .loading {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(29, 33, 37, 0.85);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            flex-direction: column;
        }

        .loading-content {
            background-color: var(--ds-surface-overlay, #22272B);
            border-radius: 6px;
            padding: 24px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            max-width: 400px;
            width: 90%;
        }

        .loading-spinner {
            border: 2px solid var(--ds-border, #404040);
            border-top: 2px solid var(--ds-background-brand-bold, #0052CC);
            border-radius: 50%;
            width: 24px;
            height: 24px;
            margin: 0 auto 8px;
            animation: spin 1s linear infinite, colorPulse 2s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes colorPulse {
            0% { border-top-color: #0052CC; }
            25% { border-top-color: #FF6F61; }
            50% { border-top-color: #FFC107; }
            75% { border-top-color: #28A745; }
            100% { border-top-color: #0052CC; }
        }

        .about-section {
            background: var(--ds-surface-overlay, #22272B);
            border-radius: 3px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--ds-border, #404040);
        }

        .about-section h2 {
            color: var(--ds-text-selected, #DEEBFF);
            font-size: 20px;
            font-weight: 500;
            margin-bottom: 16px;
        }

        .about-section p {
            margin-bottom: 16px;
            line-height: 1.3;
        }

        .improved-prompt {
            background: var(--ds-surface-overlay, #22272B);
            border-radius: 3px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--ds-border-success, #57D9A3);
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 24px;
            margin-bottom: 24px;
        }

        .info-card {
            background: var(--ds-surface-overlay, #22272B);
            border-radius: 3px;
            padding: 24px;
            border: 1px solid var(--ds-border, #404040);
            overflow-wrap: break-word;
            word-wrap: break-word;
            max-width: 100%;
        }

        .info-card .content {
            overflow-wrap: break-word;
            word-wrap: break-word;
            max-width: 100%;
        }

        .card-title {
            color: var(--ds-text-selected, #DEEBFF);
            font-size: 16px;
            font-weight: 500;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-title::before {
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: var(--ds-background-brand-bold, #0052CC);
            border-radius: 50%;
        }

        .content {
            white-space: pre-wrap;
            line-height: 1.5;
            font-size: 14px;
            overflow-wrap: break-word;
            word-wrap: break-word;
            max-width: 100%;
        }

        .back-link {
            display: inline-flex;
            align-items: center;
            color: var(--ds-text-link, #4C9AFF);
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
        }

        .back-link:hover {
            color: var(--ds-text-link-hovered, #579DFF);
            text-decoration: underline;
        }

        .back-link::before {
            content: '←';
            margin-right: 8px;
        }

        .conversation-area {
            border: 2px solid var(--ds-border-input, #404040);
            border-radius: 3px;
            background-color: var(--ds-surface-sunken, #161A1D);
            padding: 16px;
            height: 300px;
            overflow-y: auto;
            margin-bottom: 8px;
            display: flex;
            flex-direction: column;
        }
        
        .conversation-message {
            margin-bottom: 12px;
            padding: 8px 12px;
            border-radius: 4px;
            max-width: 85%;
            word-wrap: break-word;
        }
        
        .system {
            background-color: var(--ds-surface-overlay, #22272B);
            align-self: flex-start;
        }
        
        .user {
            background-color: var(--ds-background-brand-bold, #0052CC);
            color: white;
            align-self: flex-end;
            margin-left: auto;
        }
        
        .input-container {
            display: flex;
            padding: 0;
            border: none;
        }
        
        .user-input {
            flex-grow: 1;
            margin-bottom: 0;
            border-radius: 3px 0 0 3px;
        }
        
        .send-button {
            width: auto;
            padding: 8px 16px;
            border-radius: 0 3px 3px 0;
            margin: 0;
        }
        
        .toggle-container {
            margin-bottom: 16px;
        }
        
        .toggle {
            display: inline-flex;
            align-items: center;
            cursor: pointer;
        }
        
        .toggle input {
            display: none;
        }
        
        .toggle-slider {
            position: relative;
            width: 36px;
            height: 20px;
            background-color: var(--ds-surface-overlay, #22272B);
            border-radius: 10px;
            margin-right: 8px;
            transition: background-color 0.2s ease;
        }
        
        .toggle-slider:before {
            content: '';
            position: absolute;
            height: 16px;
            width: 16px;
            left: 2px;
            bottom: 2px;
            background-color: var(--ds-text-subtle, #9FADBC);
            border-radius: 50%;
            transition: transform 0.2s ease;
        }
        
        .toggle input:checked + .toggle-slider {
            background-color: var(--ds-background-brand-bold, #0052CC);
        }
        
        .toggle input:checked + .toggle-slider:before {
            transform: translateX(16px);
            background-color: white;
        }
        
        .toggle-label {
            font-size: 14px;
            color: var(--ds-text-subtle, #9FADBC);
        }
        
        .thinking-text {
            animation: pulse 1.5s infinite ease-in-out;
        }
        
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }

        hr {
            border: none;
            border-top: 1px solid var(--ds-border, #404040);
            margin: 16px 0;
            max-width: 100%;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-content">
            <div class="nav-brand"><span>de</span>Prompt</div>
            <div class="nav-links">
                <a href="/" class="nav-link {{ home_active }}">Prompt Improver</a>
                <a href="/about" class="nav-link {{ about_active }}">About</a>
            </div>
        </div>
    </nav>
    
    <!-- Main content container (to be updated dynamically) -->
    <div class="page-container" id="pageContainer">
        {{ content | safe }}
    </div>

    <!-- Loader with dynamic messages -->
    <div class="loading" id="loadingIndicator">
        <div class="loading-content">
        <div class="loading-spinner"></div>
            <p id="loaderMessage">dePrompt is enhancing your prompt...</p>
        </div>
    </div>

    <!-- AJAX submission & dynamic loader script -->
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const messages = [
            "dePrompt is enhancing your prompt...",
            "Still brewing the magic...",
            "Your prompt is being refined...",
            "Almost there... dePrompt is working its magic...",
            "Patience! Creating the perfect prompt..."
        ];
        const loaderMessage = document.getElementById('loaderMessage');
        let messageIndex = 0;
        let interval = null;

        // Track page views and start time tracking
        gtag('event', 'page_view', {
            page_title: document.title,
            page_location: window.location.href
        });

        // Track time spent on page
        let startTime = Date.now();
        window.addEventListener('beforeunload', function() {
            const timeSpent = Math.round((Date.now() - startTime) / 1000); // Convert to seconds
            gtag('event', 'time_spent', {
                time_spent_seconds: timeSpent,
                page_title: document.title
            });
        });

        // Track prompts per session
        let promptsInSession = 0;
        const form = document.getElementById('improveForm');
        if (form) {
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                promptsInSession++;
                
                // Track form submission
                gtag('event', 'prompt_enhancement_started', {
                    target_model: document.getElementById('target_model').value,
                    has_context: document.getElementById('context-toggle').checked,
                    prompts_in_session: promptsInSession
                });

                // Show loader and start cycling messages
                document.getElementById('loadingIndicator').style.display = 'flex';
                interval = setInterval(function() {
                    messageIndex = (messageIndex + 1) % messages.length;
                    loaderMessage.textContent = messages[messageIndex];
                }, 3000);

                // Prepare form data and send AJAX request
                const formData = new FormData(form);
                fetch(form.action, {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Track successful enhancement
                    gtag('event', 'prompt_enhancement_completed', {
                        target_model: document.getElementById('target_model').value,
                        has_context: document.getElementById('context-toggle').checked,
                        prompts_in_session: promptsInSession,
                        success: true
                    });

                    // Stop loader and update content
                    clearInterval(interval);
                    document.getElementById('loadingIndicator').style.display = 'none';
                    document.getElementById('pageContainer').innerHTML = data.html;
                })
                .catch(err => {
                    // Track error
                    gtag('event', 'prompt_enhancement_error', {
                        target_model: document.getElementById('target_model').value,
                        error_message: err.message,
                        prompts_in_session: promptsInSession,
                        success: false
                    });

                    clearInterval(interval);
                    document.getElementById('loadingIndicator').style.display = 'none';
                    alert("An error occurred while processing your request.");
                });
            });
        }

        // Track model selection changes
        const modelSelect = document.getElementById('target_model');
        if (modelSelect) {
            modelSelect.addEventListener('change', function() {
                gtag('event', 'model_selected', {
                    model: this.value,
                    prompts_in_session: promptsInSession
                });
            });
        }

        // Track context toggle
        const contextToggle = document.getElementById('context-toggle');
        if (contextToggle) {
            contextToggle.addEventListener('change', function() {
                gtag('event', 'context_mode_toggled', {
                    enabled: this.checked,
                    prompts_in_session: promptsInSession
                });
            });
        }

        // Enhanced conversation tracking
        let conversationStartTime = null;
        let messageCount = 0;
        const sendButton = document.getElementById('send-button');
        if (sendButton) {
            sendButton.addEventListener('click', function() {
                if (!conversationStartTime) {
                    conversationStartTime = Date.now();
                    gtag('event', 'conversation_started', {
                        prompts_in_session: promptsInSession
                    });
                }
                
                messageCount++;
                const userInput = document.getElementById('user-input');
                const hasInput = userInput.value.trim().length > 0;
                
                gtag('event', 'conversation_message_sent', {
                    has_input: hasInput,
                    message_count: messageCount,
                    prompts_in_session: promptsInSession
                });

                // Track conversation engagement
                if (messageCount >= 3) {
                    const conversationDuration = Math.round((Date.now() - conversationStartTime) / 1000);
                    gtag('event', 'conversation_engaged', {
                        duration_seconds: conversationDuration,
                        message_count: messageCount,
                        prompts_in_session: promptsInSession
                    });
                }
            });
        }
    });
    </script>
</body>
</html>
"""

# Route for the home page (AJAX form submission)
@app.route("/")
def index():
    content = """
        <div class="header">
            <h1><span class="brand-prefix">de</span>Prompt</h1>
            <p class="subtitle">AI Prompt Engineering Assistant</p>
        </div>

        <form id="improveForm" action="/improve_prompt" method="post">
            <div class="form-section">
                <div class="form-field">
                    <label for="target_model">Target Model</label>
                    <div class="field-description">Select the model you're writing the prompt for</div>
                    <select id="target_model" name="target_model">
                        <option value="">General Purpose</option>
                        <optgroup label="OpenAI">
                            <option value="gpt-4o">GPT-4o (128k)</option>
                            <option value="gpt-4o-mini">GPT-4o Mini (128k)</option>
                            <option value="gpt-4.5">GPT-4.5 (128k)</option>
                            <option value="o3-mini">o3-mini (200k)</option>
                            <option value="o1">o1 (200k)</option>
                        </optgroup>
                        <optgroup label="Anthropic">
                            <option value="claude-3.7-sonnet">Claude 3.7 Sonnet (200k)</option>
                            <option value="claude-3.5-sonnet">Claude 3.5 Sonnet (200k)</option>
                            <option value="claude-3-haiku">Claude 3 Haiku (200k)</option>
                        </optgroup>
                        <optgroup label="Google">
                            <option value="gemini-2.0-pro">Gemini 2.0 Pro (2M)</option>
                            <option value="gemini-2.0-flash">Gemini 2.0 Flash (1M)</option>
                            <option value="gemini-1.5-pro">Gemini 1.5 Pro (2M)</option>
                        </optgroup>
                        <optgroup label="Open Source">
                            <option value="llama-3.1-405b">Llama 3.1 405B (128k)</option>
                            <option value="llama-3.2-1b">Llama 3.2 1B (128k)</option>
                            <option value="mistral-large-2">Mistral Large 2 (32k)</option>
                        </optgroup>
                    </select>
                </div>

                <div class="form-field">
                    <label for="prompt">Original Prompt</label>
                    <div class="field-description">Enter your current prompt for improvement</div>
                    <textarea id="prompt" name="prompt" placeholder="Enter your original prompt here..." required></textarea>
                </div>

                <div id="context-container">
                    <div class="form-field">
                        <label for="context-toggle">Context</label>
                        <div class="field-description">Provide information about your use case to get better improvements</div>
                        <div class="toggle-container">
                            <label class="toggle">
                                <input type="checkbox" id="context-toggle">
                                <span class="toggle-slider"></span>
                                <span class="toggle-label">Enable context mode</span>
                            </label>
                        </div>
                        <div id="conversation-interface" style="display: none; margin-top: 16px;">
                            <div id="conversation-area" class="conversation-area">
                                <!-- Conversation messages will be added here dynamically -->
                            </div>
                            <div class="input-container">
                                <input type="text" id="user-input" placeholder="Type your response here..." class="user-input">
                                <button type="button" id="send-button" class="send-button">Send</button>
                            </div>
                        </div>
                        <textarea id="context" name="context" placeholder="Describe where and how this prompt will be used..." style="display: none;"></textarea>
                    </div>
                </div>
            </div>

            <button type="submit" id="enhance-button">Enhance Prompt</button>
        </form>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const conversationArea = document.getElementById('conversation-area');
                const userInput = document.getElementById('user-input');
                const sendButton = document.getElementById('send-button');
                const contextField = document.getElementById('context');
                const enhanceButton = document.getElementById('enhance-button');
                const contextToggle = document.getElementById('context-toggle');
                const conversationInterface = document.getElementById('conversation-interface');
                
                // Store conversation history
                let conversationHistory = [];
                let questionGenerated = false;
                
                // Toggle conversation interface
                contextToggle.addEventListener('change', function() {
                    if (this.checked) {
                        conversationInterface.style.display = 'block';
                        contextField.style.display = 'none';
                        
                        // Initialize conversation if it's the first time
                        if (!questionGenerated) {
                            // Generate the first question using the model
                            generateNextQuestion();
                        }
                    } else {
                        conversationInterface.style.display = 'none';
                        contextField.style.display = 'block';
                    }
                });
                
                // Enable the enhance button initially since context is optional
                enhanceButton.disabled = false;
                
                // Function to generate the next question using the model
                function generateNextQuestion() {
                    // Get the original prompt text
                    const originalPrompt = document.getElementById('prompt').value.trim();
                    
                    // Show loading indicator in conversation area
                    const thinkingMessage = document.createElement('div');
                    thinkingMessage.classList.add('conversation-message', 'system', 'thinking-text');
                    thinkingMessage.textContent = "Thinking of a relevant question...";
                    conversationArea.appendChild(thinkingMessage);
                    conversationArea.scrollTop = conversationArea.scrollHeight;
                    
                    // Add to history (will be removed after response)
                    conversationHistory.push({
                        role: 'system',
                        content: "Thinking of a relevant question..."
                    });
                    
                    // Make AJAX request to generate question
                    fetch('/generate_question', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            conversation: conversationHistory,
                            target_model: document.getElementById('target_model').value,
                            original_prompt: originalPrompt
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Remove the loading message
                        conversationArea.removeChild(conversationArea.lastChild);
                        conversationHistory.pop();
                        
                        if (data.question) {
                            // Add the generated question
                            addMessage(data.question, "system");
                            questionGenerated = true;
                        } else if (data.complete) {
                            // All questions complete - model has enough information
                            enhanceButton.disabled = false;
                            addMessage("I have enough information now! You can submit your prompt for enhancement.", "system");
                            
                            // Disable the input and send button
                            userInput.disabled = true;
                            sendButton.disabled = true;
                            userInput.placeholder = "Conversation complete";
                        } else {
                            // Error occurred
                            addMessage("I couldn't generate a question. Please provide any additional context you think is relevant.", "system");
                        }
                    })
                    .catch(error => {
                        // Remove the loading message
                        conversationArea.removeChild(conversationArea.lastChild);
                        conversationHistory.pop();
                        
                        // Add error message
                        addMessage("Sorry, there was an error generating a question. Please try again or provide context directly in the text area.", "system");
                        console.error('Error:', error);
                    });
                }
                
                // Function to add a message to the conversation
                function addMessage(text, sender) {
                    const messageDiv = document.createElement('div');
                    messageDiv.classList.add('conversation-message', sender);
                    messageDiv.textContent = text;
                    conversationArea.appendChild(messageDiv);
                    conversationArea.scrollTop = conversationArea.scrollHeight;
                    
                    // Store in history
                    conversationHistory.push({
                        role: sender === 'user' ? 'user' : 'system',
                        content: text
                    });
                    
                    // Update hidden context field with formatted conversation
                    updateContextField();
                }
                
                // Function to update the hidden context field
                function updateContextField() {
                    let formattedContext = '';
                    for (let i = 0; i < conversationHistory.length; i += 2) {
                        if (i < conversationHistory.length) {
                            const question = conversationHistory[i].content;
                            const answer = i + 1 < conversationHistory.length ? conversationHistory[i + 1].content : '';
                            formattedContext += `Q: ${question}\nA: ${answer}\n\n`;
                        }
                    }
                    contextField.value = formattedContext.trim();
                }
                
                // Send button click handler
                sendButton.addEventListener('click', function() {
                    const text = userInput.value.trim();
                    if (text) {
                        addMessage(text, 'user');
                        userInput.value = '';
                        
                        // Generate the next question
                        generateNextQuestion();
                    }
                });
                
                // Handle Enter key press
                userInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendButton.click();
                        e.preventDefault();
                    }
                });
            });
        </script>
    """
    return render_template_string(BASE_TEMPLATE, content=content, home_active="active", about_active="")

# About page (unchanged)
@app.route("/about")
def about():
    content = """
        <style>
            .about-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 24px;
            }
            
            .about-card {
                background: var(--ds-surface-overlay, #22272B);
                border-radius: 6px;
                padding: 20px;
                border: 1px solid var(--ds-border, #404040);
            }
            
            .about-card-title {
                display: flex;
                align-items: center;
                margin-bottom: 12px;
                color: var(--ds-text-selected, #DEEBFF);
                font-size: 16px;
                font-weight: 500;
            }
            
            .about-icon {
                width: 20px;
                height: 20px;
                margin-right: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: var(--ds-background-brand-bold, #0052CC);
            }
            
            .about-card-title .brand-prefix {
                margin-right: 0;
                margin-left: 4px;
            }
            
            .about-card p {
                margin: 0;
            }
        </style>
        
        <div class="about-grid">
            <div class="about-card">
                <div class="about-card-title">
                    <div class="about-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                        </svg>
                    </div>
                    About <span class="brand-prefix">de</span>Prompt
                </div>
                <p><span class="brand-prefix">de</span>Prompt optimizes prompts. AI often misunderstands you. Good prompts fix that.</p>
            </div>
            
            <div class="about-card">
                <div class="about-card-title">
                    <div class="about-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z"/>
                        </svg>
                    </div>
                    Why <span class="brand-prefix">de</span>Prompt?
                </div>
                <p>Without context, AI guesses what you want. With context, it knows. Simple as that.</p>
            </div>
            
            <div class="about-card">
                <div class="about-card-title">
                    <div class="about-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                            <path d="M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5z"/>
                        </svg>
                    </div>
                    Who needs <span class="brand-prefix">de</span>Prompt?
                </div>
                <p>You do. If you build with AI, write with AI, or design with AI.</p>
            </div>
            
            <div class="about-card">
                <div class="about-card-title">
                    <div class="about-icon">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
                            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                        </svg>
                    </div>
                    Made by
                </div>
                <p>Purav Bhardwaj. Find him on <a href="https://atlassian.enterprise.slack.com/team/U02EVDVK1BK">Slack</a>.</p>
            </div>
        </div>
    """
    return render_template_string(BASE_TEMPLATE, content=content, home_active="", about_active="active")

# AJAX endpoint for prompt improvement
@app.route("/improve_prompt", methods=["POST"])
def improve_prompt():
    # Simulate delay for testing so the loader cycles through messages
    time.sleep(3)
    
    context = request.form.get("context", "")
    original_prompt = request.form.get("prompt", "")
    target_model = request.form.get("target_model", "")

    # Analyze the context for requirements and quality factors
    context_analysis = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": """You are an expert at analyzing technical contexts and requirements.
Analyze the given context and prompt to extract key requirements, constraints, and objectives.
Format your response as JSON with the following structure:
{
    "domain": "technical domain of use case",
    "critical_requirements": ["list of must-have requirements"],
    "constraints": ["list of limitations or constraints"],
    "success_criteria": ["list of what makes a good response"],
    "risk_factors": ["potential issues to address"],
    "complexity_level": "low|medium|high",
    "format_type": "coding|creative|analytical|conversational|academic|other",
    "format_requirements": ["specific formatting needs for this domain"],
    "confidence_score": 0.0-1.0
}

For the "format_type" field, choose the most appropriate value:
- "coding": For programming, technical documentation, or algorithmic tasks
- "creative": For storytelling, content creation, or artistic prompts
- "analytical": For data analysis, research, or problem-solving tasks
- "conversational": For chatbots, customer service, or dialogue-based tasks
- "academic": For educational, scholarly, or formal writing tasks
- "other": For any domain not fitting above (specify details in format_requirements)

For "format_requirements", include specific formatting needs like:
- Use of technical language or terminology
- Need for examples or demonstrations
- Necessary structure (lists, paragraphs, code blocks)
- Length considerations (concise vs. detailed)
- Tone and style requirements"""},
            {"role": "user", "content": f"Context: {context}\nOriginal Prompt: {original_prompt}\n\nAnalyze this context and prompt to understand the requirements and constraints."}
        ]
    )

    # Parse and trim context analysis output
    analysis = json.loads(context_analysis.choices[0].message.content)
    analysis['domain'] = analysis['domain'].strip()
    analysis['complexity_level'] = analysis['complexity_level'].strip()
    analysis['critical_requirements'] = [req.strip() for req in analysis['critical_requirements']]
    analysis['constraints'] = [c.strip() for c in analysis.get('constraints', [])]
    analysis['success_criteria'] = [s.strip() for s in analysis.get('success_criteria', [])]
    analysis['risk_factors'] = [r.strip() for r in analysis.get('risk_factors', [])]
    analysis['format_type'] = analysis.get('format_type', 'other').strip()
    analysis['format_requirements'] = [f.strip() for f in analysis.get('format_requirements', [])]

    # Prepare system message with explicit instructions for output sections
    system_message = f"""You are an expert prompt engineer with deep understanding of LLM capabilities and limitations.

CONTEXT ANALYSIS:
Domain: {analysis['domain']}
Critical Requirements: {', '.join(analysis['critical_requirements'])}
Constraints: {', '.join(analysis['constraints'])}
Success Criteria: {', '.join(analysis['success_criteria'])}
Risk Factors: {', '.join(analysis['risk_factors'])}
Complexity: {analysis['complexity_level']}
Format Type: {analysis['format_type']}
Format Requirements: {', '.join(analysis['format_requirements'])}

TARGET MODEL CHARACTERISTICS:
{model_guidance.get(target_model, {}).get('considerations', 'General purpose model')}

PROMPT FORMAT GUIDANCE:
Based on the domain "{analysis['domain']}", format type "{analysis['format_type']}", and complexity level "{analysis['complexity_level']}", adjust your output format:

1. For coding/technical tasks:
   - Use concise, precise language
   - Include code-like structure with clear delimiters
   - Specify input/output formats explicitly
   - Use markdown code blocks for examples

2. For creative writing tasks:
   - Use more open-ended framing
   - Include inspirational elements
   - Balance constraints with creative freedom
   - Avoid overly rigid structure

3. For analytical/data tasks:
   - Use clear step-by-step guidance
   - Specify data handling expectations
   - Include validation checkpoints
   - Structure with numbered lists for clarity

4. For conversational/customer service tasks:
   - Use concise dialogue-like format
   - Include tone and style guidance
   - Structure with clear scenarios
   - Minimize verbose explanatory text

5. For academic/research tasks:
   - Use precise terminology
   - Structure with clear research methodology
   - Include citation/reference expectations
   - Balance detail with clarity

Your task is to substantially improve the provided prompt by incorporating:

1. STRUCTURAL ELEMENTS:
   - Clear organization appropriate to the domain
   - Format matching the task's nature (avoid one-size-fits-all structures)
   - Context-appropriate framing that matches actual use case
   - Output format instructions tailored to the specific domain

2. PRECISION TECHNIQUES:
   - Domain-specific constraints and terminology
   - Specific examples demonstrating expected outputs for this context
   - Quantitative metrics where applicable to this domain
   - Clear success criteria definition relevant to this use case

3. MODEL-SPECIFIC OPTIMIZATIONS:
   - {model_guidance.get(target_model, {}).get('considerations', 'General model guidance')}
   - Appropriate depth based on model capabilities
   - Strategic use of few-shot examples if needed
   - Memory management for context window limitations

4. GUARDRAILS:
   - Error prevention instructions
   - Explicit handling of edge cases
   - Confidence scoring or uncertainty indicators
   - Fallback mechanisms when appropriate

**IMPORTANT FORMATTING INSTRUCTIONS:**
1. Match your formatting to the domain and purpose
2. Avoid unnecessarily verbose or essay-like structures
3. Use formatting (bullet points, numbering, etc.) appropriate to the context
4. For simple prompts, keep enhancements simple and focused
5. For complex prompts, use appropriate structure without overcomplicating

**IMPORTANT:** Your response must include the following sections exactly as shown below:

[Improved Prompt]
...Your improved prompt here...

---
[Explanation of Changes]
...Detailed explanation of what was changed and why...

---
[Additional Considerations]
...Any further notes or fallback recommendations...

Your improved prompt must be clearly superior to the original in structure, clarity, specificity, and alignment with the target model's capabilities while maintaining an appropriate format for the specific use case.
"""
    improvement_response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Original Prompt: {original_prompt}"}
        ]
    )

    # Split the response into sections
    sections = improvement_response.choices[0].message.content.split("---")
    improved_prompt = sections[0].strip().replace("[Improved Prompt]", "").strip() if len(sections) >= 1 else "No improved prompt provided."
    
    # If there's no explanation provided, generate one
    explanation = sections[1].strip().replace("[Explanation of Changes]", "").strip() if len(sections) >= 2 and sections[1].strip() else ""
    
    if not explanation or explanation == "No explanation provided.":
        explanation_response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": """You are an expert prompt engineer tasked with explaining improvements made to AI prompts.
                
Analyze the original and improved prompts, then provide a detailed explanation of the changes and their benefits.
Focus on:
1. Structural improvements
2. Clarity enhancements
3. Added specificity and constraints
4. Model-specific optimizations
5. Guardrails and error prevention

Your explanation should be thorough yet concise, highlighting the most significant improvements and their expected impact."""},
                {"role": "user", "content": f"Original Prompt:\n{original_prompt}\n\nImproved Prompt:\n{improved_prompt}\n\nExplain the key improvements made and why they matter."}
            ]
        )
        explanation = explanation_response.choices[0].message.content.strip()
    
    # If there's no additional considerations provided, generate them
    considerations = sections[2].strip().replace("[Additional Considerations]", "").strip() if len(sections) >= 3 and sections[2].strip() else ""
    
    if not considerations or considerations == "No additional considerations provided.":
        considerations_response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": """You are an expert prompt engineer tasked with providing additional considerations for AI prompts.
                
After reviewing the original and improved prompts, provide specific additional considerations that might help the user.
Focus on:
1. Alternative approaches that could be considered
2. Potential limitations of the improved prompt
3. Model-specific adaptations for different AI models
4. Testing recommendations to validate effectiveness
5. Fallback strategies for edge cases

Your considerations should be practical, specific, and immediately useful to the prompt user."""},
                {"role": "user", "content": f"Original Prompt:\n{original_prompt}\n\nImproved Prompt:\n{improved_prompt}\n\nTarget Model: {target_model}\n\nProvide additional considerations that would be helpful."}
            ]
        )
        considerations = considerations_response.choices[0].message.content.strip()

    # Enhanced validation for improved quality assessment
    validation_response = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": """Conduct a rigorous evaluation of the improved prompt against the original prompt and requirements.
            
Assessment criteria (rate each on scale of 1-10):

1. COMPLETENESS
   - Are all user requirements addressed?
   - Are all edge cases covered?
   - Is there sufficient context included?

2. CLARITY & STRUCTURE
   - Is the prompt clearly organized?
   - Are instructions unambiguous?
   - Is appropriate formatting used?
   - Does the structure match the domain and use case?

3. PRECISION & SPECIFICITY
   - Are constraints clearly defined?
   - Are success criteria explicit?
   - Are examples included where helpful?
   - Is domain-specific terminology used appropriately?

4. MODEL APPROPRIATENESS
   - Does it match target model capabilities?
   - Is context length optimized?
   - Are model-specific techniques used?

5. CONTEXTUAL FIT
   - How well does the format match the domain?
   - Is the style appropriate (not too verbose/essay-like)?
   - Is the complexity appropriate to the task?
   - Would this prompt work well in actual use?

6. IMPROVEMENT DELTA
   - How significant is the improvement?
   - What key weaknesses were addressed?
   - What metrics would likely improve?

Provide a final confidence score (0-1) with two decimal precision and specific recommendations for any remaining improvements.

IMPORTANT: If the enhanced prompt is overly verbose, too essay-like, or uses a structure inappropriate for the domain, flag this issue and suggest specific improvements."""},
            {"role": "user", "content": f"Original Context: {context}\nOriginal Prompt: {original_prompt}\nImproved Prompt: {improved_prompt}\nRequirements: {json.dumps(analysis)}\nTarget Model: {target_model}\nDomain: {analysis['domain']}\n\nConduct a comprehensive evaluation."}
        ]
    )
    validation_result = validation_response.choices[0].message.content

    content = f"""
        <h2 style="color: var(--ds-text-selected, #DEEBFF); margin-bottom: 32px;"><span class="brand-prefix">de</span>Prompt Results</h2>
        
        <div class="improved-prompt" style="margin-top: 16px;">
            <div class="content">{improved_prompt}</div>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <div class="card-title">Context Analysis</div>
                <div class="content">
                    <strong>Domain:</strong> {analysis['domain']}<br>
                    <strong>Format Type:</strong> {analysis['format_type']}<br>
                    <strong>Complexity:</strong> {analysis['complexity_level']}<br>
                    <strong>Critical Requirements:</strong><br>
                    {"<br>".join(f"• {req}" for req in analysis['critical_requirements'])}
                    <br><br>
                    <strong>Format Requirements:</strong><br>
                    {"<br>".join(f"• {req}" for req in analysis['format_requirements'])}
                </div>
            </div>

            <div class="info-card">
                <div class="card-title">Explanation of Changes</div>
                <div class="content">{explanation}</div>
            </div>
            
            <div class="info-card">
                <div class="card-title">Additional Considerations</div>
                <div class="content">{considerations}</div>
            </div>

            <div class="info-card">
                <div class="card-title">Validation Results</div>
                <div class="content">{validation_result}</div>
            </div>
        </div>
        
        <a href="/" class="back-link">Create Another Prompt</a>
    """

    # Return the inner content as JSON (to be injected via AJAX)
    return jsonify({"html": content})

# API endpoint for generating conversation questions
@app.route("/generate_question", methods=["POST"])
def generate_question():
    data = request.json
    conversation = data.get('conversation', [])
    target_model = data.get('target_model', '')
    original_prompt = data.get('original_prompt', '')
    
    # Determine if this is the first question or a follow-up
    is_first_question = len(conversation) == 0
    
    # Create a system prompt for the question generation
    if is_first_question:
        system_prompt = f"""You are an expert prompt engineer who helps improve AI prompts with focused questions.
Your goal: Gather essential context to improve the prompt's effectiveness.

IMPORTANT: Ask ONE critical question (10 words max) that would most improve the prompt's clarity, effectiveness, or alignment with the target model.

Original prompt:
{original_prompt}

Target model info:
{model_guidance.get(target_model, {}).get('considerations', 'General purpose model')}

Provide a single, clear, ultra-brief question focusing on the most critical missing information that would help improve the prompt's effectiveness."""
    else:
        # Extract previous questions and answers
        qa_pairs = []
        for i in range(0, len(conversation), 2):
            if i + 1 < len(conversation):
                qa_pairs.append({
                    "question": conversation[i]["content"],
                    "answer": conversation[i + 1]["content"]
                })
        
        # Analyze what we know and what we need to ask next
        system_prompt = f"""You are an expert prompt engineer focused on gathering comprehensive context.
Your goal: Ensure we have all necessary information to improve the prompt effectively.

Original prompt:
{original_prompt}

Conversation so far:
{json.dumps(qa_pairs, indent=2)}

Target model info:
{model_guidance.get(target_model, {}).get('considerations', 'General purpose model')}

Instructions:
1. Review the conversation history and determine if we have gathered enough context to make meaningful prompt improvements.
2. Consider these key areas:
   - Use case and intended audience
   - Specific requirements and constraints
   - Success criteria and quality expectations
   - Error handling and edge cases
   - Model-specific considerations
   - Performance and efficiency needs
   - Security and compliance requirements
   - Integration and compatibility needs

3. If ANY of these areas are unclear or missing, ask ONE ultra-brief question (10 words max) about the most critical missing information.
4. Only respond with "COMPLETE" if we have gathered sufficient information across ALL key areas.

Remember: It's better to ask one more question than to miss critical context."""

    try:
        # Call the OpenAI API to generate the next question
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate the next question to ask."}
            ]
        )
        
        question = response.choices[0].message.content.strip()
        
        # Check if we have enough information
        if question.upper() == "COMPLETE":
            return jsonify({"complete": True})
        
        return jsonify({"question": question})
    except Exception as e:
        print(f"Error generating question: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Get port from environment variable for production
    port = int(os.environ.get("PORT", 8000))
    # Start the Flask server
    app.run(host='0.0.0.0', port=port, debug=False)
