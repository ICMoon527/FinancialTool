# -*- coding: utf-8 -*-
"""
LLM 连通性测试脚本

用于验证 .env 中配置的 AI 大模型是否可用。
发送一个简单问题 "你是谁？" 并打印模型返回的答案。
"""

import sys
from pathlib import Path

# 将项目根目录添加到 sys.path，确保可以 import 项目模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_llm() -> None:
    """测试 LLM 连通性并打印结果。"""
    from src.config import setup_env, get_config

    # 1. 加载 .env 配置
    setup_env()
    config = get_config()

    # 2. 检查配置
    model = config.litellm_model
    if not model:
        print("❌ 未配置 LITELLM_MODEL，请在 .env 中设置 LITELLM_MODEL 或 GEMINI_API_KEY / OPENAI_API_KEY 等")
        sys.exit(1)

    # 根据 model 前缀获取对应的 API keys
    from src.analyzer import GeminiAnalyzer
    keys = GeminiAnalyzer._get_api_keys_for_model(model, config)
    extra_params = GeminiAnalyzer._extra_litellm_params(model, config)

    if not keys:
        print(f"❌ 模型 {model} 对应的 API Key 未配置")
        if model.startswith("gemini/"):
            print("   请设置 GEMINI_API_KEY 或 GEMINI_API_KEYS")
        elif model.startswith("anthropic/"):
            print("   请设置 ANTHROPIC_API_KEY 或 ANTHROPIC_API_KEYS")
        else:
            print("   请设置 OPENAI_API_KEY 或 OPENAI_API_KEYS")
        sys.exit(1)

    print(f"🔧 模型: {model}")
    print(f"🔑 API Key 数量: {len(keys)}")
    if extra_params.get("api_base"):
        print(f"🌐 API Base: {extra_params['api_base']}")
    print(f"📤 发送测试问题: 你是谁？")
    print("-" * 50)

    # 3. 调用 LLM
    import litellm
    import time

    start = time.time()
    try:
        call_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": "你是谁？"}],
            "api_key": keys[0],
            "temperature": 0.3,
            "max_tokens": 256,
            "num_retries": 0,
        }
        call_kwargs.update(extra_params)

        response = litellm.completion(**call_kwargs)
        elapsed = time.time() - start

        reply = response.choices[0].message.content
        print(f"✅ 成功 (耗时 {elapsed:.1f}s)")
        print(f"📥 模型回答:")
        print(f"   {reply}")

    except litellm.AuthenticationError as e:
        elapsed = time.time() - start
        print(f"❌ 认证失败 (耗时 {elapsed:.1f}s)")
        print(f"   {e}")
        print("   请检查 .env 中的 API Key 是否正确")
        sys.exit(1)

    except litellm.RateLimitError as e:
        elapsed = time.time() - start
        print(f"❌ 请求频率限制 (耗时 {elapsed:.1f}s)")
        print(f"   {e}")
        sys.exit(1)

    except Exception as e:
        elapsed = time.time() - start
        print(f"❌ 调用失败 (耗时 {elapsed:.1f}s)")
        print(f"   {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    test_llm()