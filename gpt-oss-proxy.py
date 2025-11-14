#!/usr/bin/env python3
from flask import Flask, request, Response, stream_with_context
import requests
import re
import json

app = Flask(__name__)

def clean_response(text):
    """Clean GPT-OSS channel tags and format nicely"""
    # Remove channel tags entirely for cleaner output
    text = re.sub(r'<\|channel\|>analysis<\|message\|>.*?<\|end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|start\|>assistant<\|channel\|>final<\|message\|>', '', text)
    text = re.sub(r'<\|channel\|>final<\|message\|>', '', text)
    text = re.sub(r'<\|end\|>', '', text)
    return text.strip()

def stream_and_clean(response):
    """Stream response and clean channel tags on the fly"""
    buffer = ""
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
            buffer += chunk
            # Try to parse as SSE
            if chunk.startswith('data: '):
                try:
                    data_str = chunk[6:]  # Remove 'data: ' prefix
                    if data_str.strip() == '[DONE]':
                        yield chunk
                        continue

                    data = json.loads(data_str)

                    # Clean the content in the delta
                    if 'choices' in data:
                        for choice in data['choices']:
                            if 'delta' in choice and 'content' in choice['delta']:
                                choice['delta']['content'] = clean_response(choice['delta']['content'])
                            elif 'text' in choice:
                                choice['text'] = clean_response(choice['text'])

                    yield f"data: {json.dumps(data)}\n\n"
                except:
                    # Not valid JSON, pass through
                    yield chunk
            else:
                yield chunk

@app.route('/')
def index():
    """Proxy the llama.cpp web UI"""
    response = requests.get('http://localhost:8002/')
    return Response(response.content, content_type='text/html')

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Handle chat completions with streaming support"""
    data = request.json

    try:
        # Forward to actual llama.cpp server with streaming
        response = requests.post(
            'http://localhost:8002/v1/chat/completions',
            json=data,
            stream=True
        )

        # Check if response is OK
        if response.status_code != 200:
            return {'error': f'Server returned {response.status_code}'}, response.status_code

        # If streaming response
        if 'text/event-stream' in response.headers.get('content-type', ''):
            return Response(
                stream_with_context(stream_and_clean(response)),
                content_type='text/event-stream'
            )

        # If regular JSON response
        try:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                result['choices'][0]['message']['content'] = clean_response(content)
            return result
        except:
            return Response(response.content, content_type='application/json')

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy_all(path):
    """Proxy all other requests to llama.cpp"""
    url = f'http://localhost:8002/{path}'

    # Forward the request
    if request.method == 'GET':
        response = requests.get(url, params=request.args, stream=True)
    elif request.method == 'POST':
        response = requests.post(url, json=request.json, stream=True)
    elif request.method == 'PUT':
        response = requests.put(url, json=request.json, stream=True)
    elif request.method == 'DELETE':
        response = requests.delete(url, stream=True)
    elif request.method == 'PATCH':
        response = requests.patch(url, json=request.json, stream=True)

    # Handle streaming responses (SSE)
    if 'text/event-stream' in response.headers.get('content-type', ''):
        return Response(
            stream_with_context(stream_and_clean(response)),
            content_type='text/event-stream'
        )

    # Pass through other responses
    return Response(response.content, content_type=response.headers.get('content-type'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)
