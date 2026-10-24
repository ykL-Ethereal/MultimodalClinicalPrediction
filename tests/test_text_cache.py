from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np

from clinical_multimodal.config import TextConfig
from tests.helpers import DemoTextBackend
from clinical_multimodal.preprocessing.text import MacBertBackend, chunk_token_ids, clean_text
from clinical_multimodal.preprocessing.text_cache import TextCache
from tests.helpers import TestWorkspace


class CharacterTokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


class TextTests(unittest.TestCase):
    def test_cleaning_preserves_medical_meaning(self):
        text = "未见异常\r\n体温37.2℃，10mg，否认胸痛。既往2020年诊断。\x00"
        result = clean_text(text)
        self.assertEqual(result,"未见异常\n体温37.2℃，10mg，否认胸痛。既往2020年诊断。")

    def test_long_sentences_and_boundaries_preserve_all_body_tokens(self):
        tokenizer = CharacterTokenizer()
        for text in ("甲"*2000,"第一段。\n第二段！"*100,"甲"*510,"甲"*511):
            chunks = chunk_token_ids(text,tokenizer,512)
            self.assertTrue(all(0<len(x)<=510 for x in chunks))
            self.assertEqual(sum(chunks,[]),tokenizer.encode(clean_text(text)))

    def test_empty_and_special_only_note_rejected(self):
        with self.assertRaises(ValueError):
            chunk_token_ids(" \x00\n",CharacterTokenizer(),512)

    def test_cache_hit_matches_reencoding(self):
        with TestWorkspace() as tmp:
            backend = DemoTextBackend(32)
            cache = TextCache(tmp,backend)
            a = cache.get("病程文本"*100)
            b = cache.get("病程文本"*100)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(a,backend.encode("病程文本"*100))
            self.assertEqual((cache.hits,cache.misses),(1,1))
            old = cache.key("病程文本")
            for field in ("weights","tokenizer","clean","chunk","pool"):
                original = deepcopy(backend.identity)
                backend.identity[field] = "changed"
                self.assertNotEqual(old,cache.key("病程文本"))
                backend.identity = original

    def test_corrupt_cache_rejected(self):
        with TestWorkspace() as tmp:
            cache = TextCache(tmp,DemoTextBackend())
            np.save(Path(tmp)/(cache.key("test")+".npy"),np.full((1,32),np.nan))
            with self.assertRaisesRegex(ValueError,"Corrupt"):
                cache.get("test")

    def test_local_transformers_frozen_backbone_roundtrip(self):
        try:
            import torch
            from transformers import BertConfig, BertModel, BertTokenizerFast
            import safetensors
        except ImportError:
            self.skipTest("Optional integration test requires transformers and safetensors")
        with TestWorkspace() as tmp:
            root = Path(tmp)
            model_dir = root / "tiny-bert"
            model_dir.mkdir()
            tokens = ["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","未","见","异","常","。"]
            (model_dir/"vocab.txt").write_text("\n".join(tokens),encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(model_dir/"vocab.txt"))
            tokenizer.save_pretrained(model_dir)
            torch.manual_seed(42)
            model = BertModel(BertConfig(vocab_size=len(tokens),hidden_size=16,num_hidden_layers=1,
                                         num_attention_heads=2,intermediate_size=32,max_position_embeddings=512))
            model.save_pretrained(model_dir,safe_serialization=True)
            cfg = TextConfig(model_name=str(model_dir),local_files_only=True,max_tokens=16,batch_size=2)
            backend = MacBertBackend(cfg)
            a = backend.encode("未见异常。"*20)
            b = backend.encode("未见异常。"*20)
            self.assertGreater(a.shape[0],1)
            np.testing.assert_allclose(a,b,rtol=0,atol=0)
            self.assertFalse(backend.model.training)
            self.assertTrue(all(not p.requires_grad for p in backend.model.parameters()))
            backend.export(root/"snapshot")
            restored = MacBertBackend(cfg,snapshot=root/"snapshot",expected_identity=backend.identity)
            np.testing.assert_allclose(a,restored.encode("未见异常。"*20),rtol=0,atol=0)
            # Reproduce the official upstream .bin format with a bert. prefix and MLM head.
            binary_dir = root/"binary-bert"
            binary_dir.mkdir()
            tokenizer.save_pretrained(binary_dir)
            model.config.save_pretrained(binary_dir)
            state = {"bert."+k:v for k,v in model.state_dict().items()}
            state["cls.predictions.bias"] = torch.zeros(len(tokens))
            torch.save(state,binary_dir/"pytorch_model.bin")
            binary_cfg = TextConfig(model_name=str(binary_dir),local_files_only=True,max_tokens=16,batch_size=2)
            binary_backend = MacBertBackend(binary_cfg)
            np.testing.assert_allclose(a,binary_backend.encode("未见异常。"*20),rtol=0,atol=0)

