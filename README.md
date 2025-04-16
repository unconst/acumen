# Acumen

This codebase implements incentivized GRPO reinforcement learning for reasoning models on Bittensor.

## 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Create .venv
```bash
uv venv
source .venv/bin/activate
```

## 3. Install Dependencies
```bash
uv sync
```
## 4. Set your Chutes API key
    - Visit [chutes.ai](https://chutes.ai)
    - Create an account
    - Generate an API key from your account settings
    - Set your chute api env var:
```
    CHUTES_API_KEY=your_chutes_api_key
```

## 5. Set your Cloudflare R2
   - Go to Cloudflare Dashboard
   - Navigate to R2
   - Create a new bucket
   - Generate API tokens with read/write permissions
   - Set your R2 env vars:
```
    Cloudflare R2 Configuration
    R2_ACCESS_KEY_ID=your_r2_access_key
    R2_SECRET_ACCESS_KEY=your_r2_secret_key
    R2_ENDPOINT_URL=your_r2_endpoint_url
```

## 6. Run your miner or validator
```
# Validator + Miner
python3 run.py 

# Only Validator
python3 run.py --no_miner

# Only Miner
python3 run.py --no_validator
```

```
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
```