import torch
import re
import os
import json

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.pre_tokenizers import Split, Sequence
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers import Regex
from functools import lru_cache


class QwenTokenizer():
    def __init__(self, tokenizer_path, skip_special_tokens=True, char_level_chinese=True):
        super().__init__()
        # NOTE: non-chat model, all these special tokens keep randomly initialized.
        special_tokens = {
            'eos_token': '<|endoftext|>',
            'pad_token': '<|endoftext|>',
            'additional_special_tokens': [
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eom_id|>",
                "<|eot_id|>",
                "<|audio_bos|>",
                "<|audio_eos|>",
                "<|audio_out_bos|>",
                "<|AUDIO|>",
                "<|AUDIO_OUT|>",
                "<|recipient|>"
            ]
        }

        self.special_tokens = special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer_path = tokenizer_path
        self.skip_special_tokens = skip_special_tokens
        self.char_level_chinese = char_level_chinese
        
        # 中文字符的 Unicode 范围
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')
        
        if self.char_level_chinese:
            self._disable_chinese_merges()

    def _disable_chinese_merges(self):
        """屏蔽包含多个中文字符的 merge，通过重建 tokenizer.json 达到字符级别效果"""
        # 优先通过重建 tokenizer.json 的 merges 实现，避免直接操作后端属性不可见的问题
        self._rebuild_tokenizer_with_filtered_merges()

    def _rebuild_tokenizer_with_filtered_merges(self):
        """使用 tokenizer.json 过滤包含多个中文字符的 merges，并重建 Fast tokenizer"""
        import warnings
        
        try:
            json_path = os.path.join(self.tokenizer_path, "tokenizer.json")
            if not os.path.exists(json_path):
                raise FileNotFoundError(
                    f"char_level_chinese=True requires tokenizer.json, but not found at: {json_path}"
                )
                
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            model = data.get("model", {})
            merges = model.get("merges")
            vocab = model.get("vocab", {})
            
            if not (merges and vocab):
                raise ValueError(
                    f"char_level_chinese=True requires valid 'merges' and 'vocab' in tokenizer.json"
                )
            
            tok_orig = Tokenizer.from_str(json.dumps(data, ensure_ascii=False))

            def chinese_count(text: str) -> int:
                return sum(1 for c in text if self.chinese_pattern.match(c))

            def is_multi_chinese(merge_item) -> bool:
                # merge_item 可能是列表 ["a", "b"] 或字符串 "a b"
                if isinstance(merge_item, list):
                    if len(merge_item) != 2:
                        return False
                    a, b = merge_item
                elif isinstance(merge_item, str):
                    parts = merge_item.split()
                    if len(parts) != 2:
                        return False
                    a, b = parts
                else:
                    return False
                
                ida = vocab.get(a)
                idb = vocab.get(b)
                if ida is None or idb is None:
                    return False
                decoded = tok_orig.decode([ida, idb])
                return chinese_count(decoded) >= 2

            filtered_merges = [m for m in merges if not is_multi_chinese(m)]
            data["model"]["merges"] = filtered_merges

            tok = Tokenizer.from_str(json.dumps(data, ensure_ascii=False))
            # 不修改pre-tokenizer，保持原始的ByteLevel，这样单字汉字token可以直接使用
            fast = PreTrainedTokenizerFast(tokenizer_object=tok)
            fast.add_special_tokens(self.special_tokens)
            self.tokenizer = fast
        except (FileNotFoundError, ValueError) as e:
            # 明确的配置错误，直接抛出
            raise RuntimeError(
                f"Failed to enable char_level_chinese: {e}\n"
                f"char_level_chinese requires a valid tokenizer.json with BPE model."
            ) from e
        except Exception as e:
            # 其他未预期的错误，给出警告但不中断
            import warnings
            warnings.warn(
                f"Failed to enable char_level_chinese due to unexpected error: {e}\n"
                f"Falling back to original tokenizer (word-level Chinese).",
                RuntimeWarning
            )
    
    def encode(self, text, **kwargs):
        tokens = self.tokenizer([text], return_tensors="pt")
        tokens = tokens["input_ids"][0].cpu().tolist()
        return tokens

    def decode(self, tokens):
        tokens = torch.tensor(tokens, dtype=torch.int64)
        text = self.tokenizer.batch_decode([tokens], skip_special_tokens=self.skip_special_tokens)[0]
        return text


@lru_cache(maxsize=None)
def get_qwen_tokenizer(
    tokenizer_path: str,
    skip_special_tokens: bool = False,
    char_level_chinese: bool = False
) -> AutoTokenizer:
    qwen_tokenizer = QwenTokenizer(
        tokenizer_path=tokenizer_path, 
        skip_special_tokens=skip_special_tokens,
        char_level_chinese=char_level_chinese
    )
    return qwen_tokenizer.tokenizer  # qwen use AutoTokenizer interface.
