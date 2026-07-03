from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI



SUPPORTED_MODELS = ['llama', 'chatgpt', "gpt-5.2", 'o1-mini', 'gemini', 'claude', "sonnet-4.6", 'r1', 'v3', 'ChatNT']


def instantiate_models(keys: dict, 
                       models: list[str], 
                       max_tokens: int, 
                       temperature: float,
                       with_search: bool) -> dict:
    if max_tokens is None:
        max_tokens = 256
    
    chat_models = {}
    for model in models:
        if model == 'gemini':
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ImportError(
                    "Model 'gemini' requires langchain-google-genai. Install it with: pip install langchain-google-genai"
                ) from exc
            replier = ChatGoogleGenerativeAI(
                model="gemini-1.5-pro",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
                google_api_key=keys[model]
            )

        if model == 'chatgpt':
            replier = ChatOpenAI(
                model='gpt-4o',
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
                api_key=keys[model]
            )
        if model == 'gpt-5.2':
            replier = ChatOpenAI(
                model='gpt-5.2',
                temperature=1, # only value supported
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
                api_key=keys[model],
                reasoning={
                    "effort":"low",
                    "summary":"auto"
                }
            )
            if with_search:
                replier = replier.bind_tools([{"type": "web_search"}])
        if model == 'o1-mini':
            replier = ChatOpenAI(
                model="o1-mini",
                api_key=keys[model],
                temperature=1, # only param supported
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2
            )
        if model == 'claude':
            replier = ChatAnthropic(
                model='claude-3-5-sonnet-latest',
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=2,
                timeout=None,
                api_key=keys[model]
            )
        if model == 'sonnet-4.6':
            replier = ChatAnthropic(
                model='claude-opus-4-5-20251101',
                temperature=1, # only param supported
                max_tokens=max_tokens*1.5,
                max_retries=2,
                timeout=None,
                api_key=keys[model],
                thinking={"type": "enabled", "budget_tokens": int(max_tokens * 0.75)},
                #model_kwargs={"output_config": {"effort": "low"}},
            )
            if with_search:
                replier = replier.bind_tools([
                    {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
                ])
        if model == 'llama':
            replier = ChatOpenAI(
                model="llama3.1-405b",
                api_key=keys[model],
                base_url='https://api.llama-api.com',
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2
            )
        if model == 'v3':
            replier = ChatOpenAI(
                model="deepseek-chat",
                api_key=keys[model],
                base_url='https://api.deepseek.com', 
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2
            )
        if model == 'r1':
            replier = ChatOpenAI(
                model="deepseek-reasoner",
                api_key=keys[model],
                base_url='https://api.deepseek.com', 
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=None,
                max_retries=2,
                reasoning_effort ='low'
            )
        if model == 'ChatNT':
            from transformers import pipeline
            import torch
            pipe = pipeline(
                model="InstaDeepAI/ChatNT", 
                trust_remote_code=True, 
                device='cuda', 
                torch_dtype=torch.bfloat16
            )
            pipe.max_tokens = max_tokens
            replier = pipe
        if model not in SUPPORTED_MODELS:
            raise NameError(f'Model:{model} not implemented for answering DB.')
        chat_models[model] = replier

    return chat_models