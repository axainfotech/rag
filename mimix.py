import os
from dotenv import load_dotenv
from anthropic import Anthropic

# ── Load environment variables ───────────────────────────
load_dotenv()

BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
API_KEY  = os.getenv("ANTHROPIC_AUTH_TOKEN")
MODEL    = os.getenv("ANTHROPIC_MODEL")

print("DEBUG:")
print("BASE_URL:", BASE_URL)
print("MODEL:", MODEL)
print("API_KEY present:", bool(API_KEY))

# ── Validate config ──────────────────────────────────────
if not API_KEY:
    raise ValueError("❌ API key not loaded. Check your .env file.")

if not MODEL:
    raise ValueError("❌ Model not set. Check ANTHROPIC_MODEL in .env")

# ── Initialize Anthropic client (MiniMax backend) ────────
client = Anthropic(
    base_url=BASE_URL,
    api_key=API_KEY.strip()
)

# ── Make request ─────────────────────────────────────────
try:
    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": "Explain Kubernetes in 3 simple points"}
        ]
    )

    # ── Extract ONLY text blocks (ignore ThinkingBlock) ───
    output_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            output_text += block.text

    print("\n✅ SUCCESS RESPONSE:\n")
    print(output_text.strip())

except Exception as e:
    print("\n❌ ERROR:\n")
    print(str(e))