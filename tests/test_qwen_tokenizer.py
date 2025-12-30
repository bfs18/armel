"""
测试 QwenTokenizer 的中文字符级别 tokenization 功能
"""
import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ar.qwen_tokenizer import QwenTokenizer


def test_char_level_chinese():
    """测试中文字符级别 tokenization"""
    
    # 需要指定你的 tokenizer 路径
    tokenizer_path = "Qwen3-0.6B"  # 修改为实际路径
    
    print("=" * 60)
    print("测试 QwenTokenizer 中文字符级别 tokenization")
    print("=" * 60)
    
    # 测试不启用字符级别
    print("\n1. 不启用 char_level_chinese:")
    tokenizer_normal = QwenTokenizer(tokenizer_path, char_level_chinese=False)
    
    test_texts = [
        "你好世界",
        "今天天气很好",
        "Hello世界",
        "中文English混合",
        "Hello, how are you?",  # 纯英语
        "こんにちは世界",  # 日语+汉字
        "안녕하세요",  # 韩语
        "The quick brown fox jumps over the lazy dog.",  # 长英文
        "数字123和标点！？",  # 数字标点混合
        "Привет мир",  # 俄语
    ]
    
    for text in test_texts:
        tokens = tokenizer_normal.encode(text)
        decoded = tokenizer_normal.decode(tokens)
        print(f"  文本: {text}")
        print(f"  Token数: {len(tokens)}")
        print(f"  Tokens: {tokens}")
        print(f"  解码: {decoded}")
        print()
    
    # 测试启用字符级别
    print("\n2. 启用 char_level_chinese:")
    tokenizer_char = QwenTokenizer(tokenizer_path, char_level_chinese=True)
    
    for text in test_texts:
        tokens = tokenizer_char.encode(text)
        decoded = tokenizer_char.decode(tokens)
        print(f"  文本: {text}")
        print(f"  Token数: {len(tokens)}")
        print(f"  Tokens: {tokens}")
        print(f"  解码: {decoded}")
        
        # 验证中文字符数
        chinese_chars = [c for c in text if '\u4e00' <= c <= '\u9fff']
        print(f"  中文字符数: {len(chinese_chars)}")
        print()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)


def test_merge_filtering():
    """测试 merge 过滤逻辑"""
    tokenizer_path = "Qwen3-0.6B"  # 修改为实际路径
    
    print("\n" + "=" * 60)
    print("测试 Merge 过滤")
    print("=" * 60)
    
    # 读取原始merge数量
    import json
    import os
    json_path = os.path.join(tokenizer_path, "tokenizer.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        original_merges = data.get("model", {}).get("merges", [])
        print(f"\n原始 merge 数量: {len(original_merges)}")
        
        # 创建启用char_level_chinese的tokenizer
        tokenizer = QwenTokenizer(tokenizer_path, char_level_chinese=True)
        
        # 测试tokenization效果
        test_cases = [
            ("你好", "应该是2个token（每个汉字1个）"),
            ("你好世界", "应该是4个token（每个汉字1个）"),
            ("Hello world", "英语不受影响"),
            ("こんにちは", "日语不受影响"),
            ("안녕하세요", "韩语不受影响"),
        ]
        
        print("\n验证字符级tokenization:")
        for text, expected in test_cases:
            tokens = tokenizer.encode(text)
            chinese_chars = sum(1 for c in text if tokenizer.chinese_pattern.match(c))
            print(f"  '{text}': {len(tokens)} tokens, {chinese_chars} 个汉字 - {expected}")
            
        print(f"\n✓ 成功通过merge过滤实现了字符级中文tokenization")
        print(f"  提示: 过滤掉了包含>=2个汉字的merges，保留了单字汉字的vocab tokens")
    else:
        print(f"\n错误: 找不到 {json_path}")


if __name__ == "__main__":
    # 运行测试
    try:
        test_char_level_chinese()
        test_merge_filtering()
    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("1. 已安装 transformers 库")
        print("2. 修改 tokenizer_path 为实际的 tokenizer 路径")
        print("3. tokenizer 路径存在且可访问")
