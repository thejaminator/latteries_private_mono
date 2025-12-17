# useful for pooling rate limited callers

from latteries import OpenAICaller, MultiClientCaller, CallerConfig, PooledCaller, CallerCache
from dotenv import load_dotenv
import os
from pathlib import Path
from openai import AsyncOpenAI
from latteries import AnthropicCaller, TinkerCaller


def load_pooled_caller(cache_path: str) -> MultiClientCaller:
    """Non-exhaustive list of models. For demonstration purposes.
    Simply copy and create a new function for your needs.
    """
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_org = os.getenv("OPENAI_ORGANIZATION")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    tinker_base_url = os.getenv("TINKER_BASE_URL")  # Optional

    assert anthropic_api_key, "Please provide an Anthropic API Key"
    assert openai_api_key, "Please provide an OpenAI API Key"
    assert openrouter_api_key, "Please provide an OpenRouter API Key"

    # code for rate limit pooling
    shared_cache = CallerCache(Path(cache_path))
    openai_caller = OpenAICaller(api_key=openai_api_key, organization=openai_org, cache_path=shared_cache)
    openai_caller_2 = OpenAICaller(api_key=os.getenv("OPENAI_API_KEY_2"), cache_path=shared_cache)
    openai_caller_3 = OpenAICaller(
        api_key=os.getenv("JAMES_API_KEY"),
        cache_path=shared_cache,
    )
    shared_openai_caller = PooledCaller([openai_caller, openai_caller_2, openai_caller_3])
    openrouter_caller = OpenAICaller(
        openai_client=AsyncOpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1"),
        cache_path=shared_cache,
    )
    # end of rate limit pooling

    tinker_caller = TinkerCaller(
        cache_path=shared_cache,
        base_url=tinker_base_url,
    )

    clients = [
        CallerConfig(name="gpt", caller=shared_openai_caller),
        CallerConfig(name="google", caller=openrouter_caller),
        CallerConfig(name="qwen", caller=openrouter_caller),
        CallerConfig(name="deepseek", caller=openrouter_caller),
        CallerConfig(name="mistral", caller=openrouter_caller),
        CallerConfig(name="llama", caller=openrouter_caller),
        CallerConfig(
            name="claude",
            caller=AnthropicCaller(api_key=anthropic_api_key, cache_path=shared_cache),
        ),
        CallerConfig(name="tinker", caller=tinker_caller),
    ]

    return MultiClientCaller(clients)
