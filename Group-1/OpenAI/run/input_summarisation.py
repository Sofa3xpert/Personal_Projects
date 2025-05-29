from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Instantiate client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

long_statement = None

def get_concise_rewrite(text, max_tokens):
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert rewriting assistant. Your task is to rewrite a long statement into a shorter version while "
                "preserving as much of the original vocabulary, tone, and style as possible. Do not include any phrases like "
                "'here is a summary:' or indicate that it is a summary. Simply produce a concise version of the statement as if "
                "the original author had written it more succinctly."
            )
        },
        {
            "role": "user",
            "content": text
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens
        )
        concise_text = response.choices[0].message.content.strip()
    except Exception as e:
        concise_text = f"API call failed: {e}"
    return concise_text


short_statement = get_concise_rewrite(long_statement, max_tokens=512)
