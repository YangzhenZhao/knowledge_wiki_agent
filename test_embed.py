#!/usr/bin/env python3
"""
Embedding 模型验证脚本
用于诊断 Ollama embedding 服务问题
"""
import sys
import traceback

# 加载项目配置
from config import OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL


def test_ollama_service():
    """测试 Ollama 服务是否运行"""
    print("=" * 50)
    print("1. 测试 Ollama 服务连接")
    print("=" * 50)
    print(f"   OLLAMA_BASE_URL: {OLLAMA_BASE_URL}")

    import requests
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama 服务运行正常")
            models = response.json().get('models', [])
            print(f"   已安装模型数量: {len(models)}")
            for m in models:
                print(f"   - {m.get('name')}")
            return True
        else:
            print(f"❌ Ollama 服务响应异常: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到 Ollama 服务 ({OLLAMA_BASE_URL})")
        print("   请确保 Ollama 服务已启动: ollama serve")
        return False
    except Exception as e:
        print(f"❌ 连接错误: {e}")
        return False


def test_embed_model():
    """测试 embedding 模型是否可用"""
    print("\n" + "=" * 50)
    print("2. 测试 Embedding 模型")
    print("=" * 50)
    print(f"   配置的 Embedding 模型: {OLLAMA_EMBED_MODEL}")

    import requests
    try:
        # 检查模型是否已下载
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m.get('name') for m in models]

            # 检查精确匹配或带 tag 的匹配
            model_found = False
            for name in model_names:
                if name == OLLAMA_EMBED_MODEL or name.startswith(OLLAMA_EMBED_MODEL + ":"):
                    model_found = True
                    break

            if model_found:
                print(f"✅ 模型 {OLLAMA_EMBED_MODEL} 已安装")
            else:
                print(f"❌ 模型 {OLLAMA_EMBED_MODEL} 未安装")
                print(f"   请运行: ollama pull {OLLAMA_EMBED_MODEL}")
                return False
    except Exception as e:
        print(f"❌ 检查模型失败: {e}")
        return False

    return True


def test_embedding_generation():
    """测试生成 embedding 向量"""
    print("\n" + "=" * 50)
    print("3. 测试生成 Embedding 向量")
    print("=" * 50)

    # 方法1: 使用 ollama.Client (项目使用的方式)
    print("\n方法1: 使用 ollama.Client (项目使用的方式)")
    try:
        import ollama
        print(f"   ollama 库版本: {ollama.__version__ if hasattr(ollama, '__version__') else '未知'}")

        # 创建指定 host 的客户端
        client = ollama.Client(host=OLLAMA_BASE_URL)
        print(f"   连接地址: {OLLAMA_BASE_URL}")

        test_text = "这是一个测试文本"
        print(f"   测试文本: '{test_text}'")
        print(f"   正在调用 client.embed()...")

        response = client.embed(
            model=OLLAMA_EMBED_MODEL,
            input=test_text
        )

        if 'embeddings' in response:
            embedding = response['embeddings'][0]
            print(f"✅ Embedding 生成成功!")
            print(f"   向量维度: {len(embedding)}")
            print(f"   向量前5个值: {embedding[:5]}")
            return True
        else:
            print(f"❌ 响应中没有 embeddings: {response}")
            return False

    except Exception as e:
        print(f"❌ ollama.Client.embed() 失败: {e}")
        print(traceback.format_exc())

    # 方法2: 直接调用 API
    print("\n方法2: 直接调用 Ollama API")
    try:
        import requests
        import json

        test_text = "这是一个测试文本"
        url = f"{OLLAMA_BASE_URL}/api/embed"
        payload = {
            "model": OLLAMA_EMBED_MODEL,
            "input": test_text
        }

        print(f"   请求 URL: {url}")
        print(f"   请求体: {json.dumps(payload, ensure_ascii=False)}")

        response = requests.post(url, json=payload, timeout=30)

        print(f"   响应状态码: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if 'embeddings' in result:
                embedding = result['embeddings'][0]
                print(f"✅ API Embedding 生成成功!")
                print(f"   向量维度: {len(embedding)}")
                print(f"   向量前5个值: {embedding[:5]}")
                return True
            else:
                print(f"❌ 响应中没有 embeddings: {result}")
                return False
        else:
            print(f"❌ API 请求失败: {response.status_code}")
            print(f"   响应内容: {response.text}")
            return False

    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        print(traceback.format_exc())
        return False


def test_ollama_version():
    """检查 Ollama 版本"""
    print("\n" + "=" * 50)
    print("4. 检查 Ollama 版本")
    print("=" * 50)

    import requests
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print(f"✅ Ollama 版本: {version_info.get('version', '未知')}")
            return True
        else:
            print(f"❌ 获取版本失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 获取版本失败: {e}")
        return False


def main():
    print("🔍 Ollama Embedding 模型诊断工具")
    print("=" * 50)

    results = []

    # 运行所有测试
    results.append(("Ollama 服务连接", test_ollama_service()))
    results.append(("Ollama 版本检查", test_ollama_version()))

    if results[0][1]:  # 只有服务正常才继续测试
        results.append(("Embedding 模型检查", test_embed_model()))
        results.append(("Embedding 生成测试", test_embedding_generation()))

    # 打印总结
    print("\n" + "=" * 50)
    print("📋 测试总结")
    print("=" * 50)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {name}: {status}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 所有测试通过! Embedding 服务正常工作。")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())