import argparse
import json
import re
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor # Added import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


SUPPORTED_MODELS = ['llama', 'chatgpt', "gpt-5.2", 'o1-mini', 'gemini', 'claude', "sonnet-4.6", 'r1', 'v3']


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
                replier = replier.bind_tools([{"type": "web_search_preview"}])
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
                model='claude-sonnet-4-6',
                temperature=1, # only param supported
                max_tokens=max_tokens,
                max_retries=2,
                timeout=None,
                api_key=keys[model],
                thinking={"type": "adaptive"},
                output_config={"effort": "low"}
            )
            if with_search:
                replier = replier.bind_tools([
                    {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}
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
        if model not in SUPPORTED_MODELS:
            raise NameError(f'Model:{model} not implemented for answering DB.')
        chat_models[model] = replier

    return chat_models