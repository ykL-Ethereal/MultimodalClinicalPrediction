from pathlib import Path
import hashlib
import shutil
import uuid
import numpy as np

from clinical_multimodal.config import Config, DataConfig, TextConfig, ModelConfig, TrainingConfig
from clinical_multimodal.data.fixed_adapter import FixedCohortBundle
from clinical_multimodal.preprocessing.fixed_numeric import FixedNumericPreprocessor
from clinical_multimodal.preprocessing.text import clean_text, CLEAN_VERSION, CHUNK_VERSION, POOL_VERSION
from clinical_multimodal.preprocessing.text_cache import TextCache
from tests.fixed_fixture import make_fixed


class TestWorkspace:
    __test__ = False

    def __init__(self):
        self.root = (Path(__file__).resolve().parents[1] / 'runs' / 'tests').resolve()
        self.path = self.root / uuid.uuid4().hex
        self.path.mkdir(parents=True)
        self.name = str(self.path)

    def cleanup(self):
        if self.path.resolve().parent != self.root:
            raise ValueError('Refusing to clean a path outside the test workspace')
        if self.path.exists():
            shutil.rmtree(self.path)

    def __enter__(self):
        return self.name

    def __exit__(self, *args):
        self.cleanup()


class DemoTextBackend:
    """Deterministic synthetic vectors, ONLY for pipeline tests; not a language model."""
    dimension = 32

    def __init__(self, max_tokens=512):
        self.max_tokens = max_tokens
        self.identity = {"backend": "demo", "weights": "sha256-v1", "tokenizer": "unicode-v1",
                         "dimension": self.dimension, "max_tokens": max_tokens,
                         "clean": CLEAN_VERSION, "chunk": CHUNK_VERSION, "pool": POOL_VERSION}

    def encode(self, text):
        text = clean_text(text)
        if not text:
            raise ValueError("Empty observed note after cleaning")
        vectors = []
        for start in range(0, len(text), self.max_tokens - 2):
            content = text[start:start + self.max_tokens - 2]
            digest = hashlib.sha256(content.encode("utf-8")).digest()
            vectors.append((np.frombuffer(digest, dtype=np.uint8).astype(np.float32) - 127.5) / 127.5)
        return np.stack(vectors)

    def export(self, path):
        return None


def fixture(directory):
    root = make_fixed(Path(directory) / 'data')
    bundle = FixedCohortBundle(root)
    processor = FixedNumericPreprocessor.fit(bundle, bundle.rows('train'))
    return root, bundle, processor


def tiny_text_model(directory):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    path = Path(directory) / 'tiny-bert'
    path.mkdir(exist_ok=True)
    if not (path / 'config.json').exists():
        tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '病', '情', '稳', '定', '。']
        (path / 'vocab.txt').write_text('\n'.join(tokens), encoding='utf-8')
        BertTokenizerFast(vocab_file=str(path / 'vocab.txt')).save_pretrained(path)
        BertModel(BertConfig(vocab_size=len(tokens), hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32)).save_pretrained(path)
    return str(path)


def small_config(directory):
    return Config(data=DataConfig(root=str(Path(directory) / 'data'), horizon=6),
                  text=TextConfig(model_name=tiny_text_model(directory), local_files_only=True,
                                  max_tokens=32, cache_dir=str(Path(directory) / 'cache')),
                  model=ModelConfig(embedding_dim=16, static_dim=8, tcn_channels=8, tcn_blocks=5,
                                    max_time_steps=120, lab_hidden=16, rnn_hidden=8,
                                    dropout=0, modality_dropout=0),
                  training=TrainingConfig(epochs=1, batch_size=2, patience=1, num_threads=1,
                                          output_dir=str(Path(directory) / 'run'))).validate()


def cache(directory):
    return TextCache(Path(directory) / 'cache', DemoTextBackend(32))
