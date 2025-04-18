# The MIT License (MIT)
# © 2025 Acumen

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import asyncio
import aiohttp
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load configuration from .env
# -----------------------------------------------------------------------------
load_dotenv(override=True)

CHECK_INTERVAL = 2  # seconds between block checks
MAX_RETRIES = 3
HTTP_TIMEOUT = 60  # seconds
RETRY_DELAY = 1    # seconds

CHUTES_API_KEY = None
def load_chutes_token():
    global CHUTES_API_KEY
    if CHUTES_API_KEY == None: 
        CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
        if not CHUTES_API_KEY:
            raise ValueError("CHUTES_API_KEY environment variable is not set")
    return CHUTES_API_KEY

# -----------------------------------------------------------------------------
# Helper: HTTP Request with retry support
# -----------------------------------------------------------------------------
async def http_request_with_retry(session: aiohttp.ClientSession, method: str, url: str, timeout: int = HTTP_TIMEOUT, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            async with session.request(method, url, timeout=timeout, **kwargs) as response:
                response.raise_for_status()
                data = await response.json()
                return data
        except asyncio.TimeoutError as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (attempt + 1)
                await asyncio.sleep(delay)
            else:
                raise e
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (attempt + 1)
                await asyncio.sleep(delay)
            else:
                raise e

# -----------------------------------------------------------------------------
# Query the LLM via Chutes API
# -----------------------------------------------------------------------------
async def prompt(
        prompt, 
        model: str = "unsloth/gemma-3-4b-it", 
        max_tokens: int = -1, 
        temperature: float = 0.7,
        timeout: int = -1
    ) -> str:
    if max_tokens == -1: max_tokens = 100000
    if timeout == -1: timeout = 1000000
    url = "https://llm.chutes.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {load_chutes_token()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    async with aiohttp.ClientSession() as session:
        try:
            data = await http_request_with_retry(session, "POST", url, headers=headers, json=payload, timeout = timeout)
            result = data['choices'][0]['message']['content']
            return result
        except Exception as e:
            raise e