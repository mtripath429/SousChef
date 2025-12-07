import json
import streamlit as st
from openai import OpenAI


def get_openai_client() -> OpenAI:
    api_key = None
    # Prefer Streamlit secrets if available. Support both a top-level
    # `OPENAI_API_KEY` and a `[general]` table (common pattern in README).
    try:
        # direct lookup (works if secrets.toml defines OPENAI_API_KEY at top level)
        api_key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        api_key = None

    if not api_key:
        # try attribute / mapping access for a `[general]` table
        try:
            general = None
            try:
                general = st.secrets.get("general")  # type: ignore[attr-defined]
            except Exception:
                # fall back to mapping or attribute access
                try:
                    general = st.secrets["general"]  # type: ignore[index]
                except Exception:
                    general = getattr(st.secrets, "general", None)  # type: ignore[attr-defined]

            if general:
                if isinstance(general, dict):
                    api_key = general.get("OPENAI_API_KEY")
                else:
                    api_key = getattr(general, "OPENAI_API_KEY", None)
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
    # Construct an explicit httpx client to avoid compatibility issues where
    # the SDK may pass a `proxies` kwarg that isn't supported by the
    # installed httpx version. Passing a pre-built client gives us control.
    try:
        import httpx

        http_client = httpx.Client()
    except Exception:
        http_client = None

    return OpenAI(api_key=api_key, http_client=http_client)


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
