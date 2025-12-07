import json
import streamlit as st
from openai import OpenAI


def get_openai_client() -> OpenAI:
    api_key = None
    # Prefer Streamlit secrets if available
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        pass
    if not api_key:
        # Also allow environment variable fallback
        import os
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in Streamlit secrets or environment."
        )
    return OpenAI(api_key=api_key)


def response_text(resp) -> str:
    """
    Extract plain text from an OpenAI Responses API result in a robust way
    across minor SDK variations.
    """
    # Newer SDKs provide a convenience accessor
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    # Fallback to traversing structured output
    try:
        return resp.output[0].content[0].text  # type: ignore[index]
    except Exception:
        pass
    # Chat completion style fallback (if someone switches APIs accidentally)
    try:
        return resp.choices[0].message.content  # type: ignore[attr-defined]
    except Exception:
        pass
    # Last resort: JSON dump
    return json.dumps(resp)
