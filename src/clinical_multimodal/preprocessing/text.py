"""Local-only clinical text handling and sentence-aware, lossless token chunking."""
from __future__ import annotations

from pathlib import Path
import re
import unicodedata
import numpy as np

from ..io import file_sha256, fingerprint

CLEAN_VERSION = "controls-newlines-v1"
CHUNK_VERSION = "sentence-greedy-tokens-v1"
POOL_VERSION = "body-token-mean-v1"


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(c for c in text if c in "\n\t" or unicodedata.category(c) != "Cc").strip()


def chunk_token_ids(text, tokenizer, max_tokens):
    capacity = max_tokens - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity <= 0:
        raise ValueError("No capacity left for body tokens")
    chunks, current = [], []
    for sentence in re.findall(r"[^。！？!?；;\n]*[。！？!?；;\n]|[^。！？!?；;\n]+$", clean_text(text)):
        ids = tokenizer.encode(sentence, add_special_tokens=False)
        if len(current) + len(ids) <= capacity:
            current.extend(ids)
            continue
        if current:
            chunks.append(current)
            current = []
        while len(ids) > capacity:
            chunks.append(ids[:capacity])
            ids = ids[capacity:]
        current = ids
    if current:
        chunks.append(current)
    if not chunks:
        raise ValueError("An observed note produced no body tokens after cleaning/tokenization")
    return chunks


class MacBertBackend:
    def __init__(self, config, *, snapshot=None, expected_identity=None):
        import torch
        import transformers
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from huggingface_hub import hf_hub_download
        self.torch = torch
        self.config = config
        self.device = torch.device(config.device)
        source = str(snapshot or config.model_name)
        local = Path(source).is_dir()
        revision = None if local else config.revision
        local_only = bool(snapshot) or config.local_files_only
        model_config = AutoConfig.from_pretrained(source, revision=revision,
                          local_files_only=local_only, trust_remote_code=False)
        resolved = None if local else getattr(model_config, "_commit_hash", None)
        if not local and not resolved:
            raise ValueError("Could not resolve an immutable model revision")
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=resolved,
                           local_files_only=local_only, trust_remote_code=False)
        local_safe = local and ((Path(source)/"model.safetensors").exists() or
                                (Path(source)/"model.safetensors.index.json").exists())
        if local_safe:
            self.model = AutoModel.from_pretrained(source, config=model_config,
                           local_files_only=True, trust_remote_code=False, use_safetensors=True)
        else:
            # The official hfl/macbert main revision publishes pytorch_model.bin,
            # not safetensors. Read ONLY tensor weights; do not require an auto-conversion PR.
            weights_path = Path(source)/"pytorch_model.bin" if local else Path(hf_hub_download(
                source, "pytorch_model.bin", revision=resolved, local_files_only=local_only))
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
                raise ValueError("Expected a plain tensor state dictionary for the text backbone")
            kwargs = {"add_pooling_layer": False} if model_config.model_type == "bert" else {}
            self.model = AutoModel.from_config(model_config, trust_remote_code=False, **kwargs)
            expected = self.model.state_dict()
            prefix = self.model.base_model_prefix + "."
            backbone = {}
            for key, value in state.items():
                name = key[len(prefix):] if key.startswith(prefix) else key
                if name in expected:
                    backbone[name] = value
                elif not (name.startswith(("cls.", "pooler.")) or name.endswith(("position_ids", "token_type_ids"))):
                    raise ValueError(f"Unexpected text checkpoint key: {key}")
            self.model.load_state_dict(backbone, strict=True)
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        self.dimension = int(self.model.config.hidden_size)
        if config.max_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Text max_tokens exceeds the backbone position limit")
        self.max_tokens = config.max_tokens
        if expected_identity is not None:
            # Snapshot files are verified against the artifact manifest before this path.
            self.identity = expected_identity
            expected_versions = {"clean": CLEAN_VERSION, "chunk": CHUNK_VERSION, "pool": POOL_VERSION}
            if any(self.identity.get(k) != v for k, v in expected_versions.items()):
                raise ValueError("Artifact text processing versions differ from this implementation")
            if self.identity["dimension"] != self.dimension or self.identity["max_tokens"] != self.max_tokens:
                raise ValueError("Text snapshot identity mismatch")
        else:
            commit = resolved
            if local:
                files = list(Path(source).glob("*.safetensors")) + list(Path(source).glob("pytorch_model*.bin"))
                weights = {p.name: file_sha256(p) for p in sorted(files)}
                commit = fingerprint(weights)
            if not commit:
                raise ValueError("Could not resolve an immutable model revision")
            self.identity = {"backend": "macbert", "model": config.model_name, "weights": commit,
                             "tokenizer": fingerprint(self.tokenizer.backend_tokenizer.to_str()),
                             "transformers": transformers.__version__, "dimension": self.dimension,
                             "max_tokens": self.max_tokens, "clean": CLEAN_VERSION,
                             "chunk": CHUNK_VERSION, "pool": POOL_VERSION}

    def encode(self, text):
        torch = self.torch
        chunks = chunk_token_ids(text, self.tokenizer, self.max_tokens)
        vectors = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(chunks), self.config.batch_size):
                encoded = []
                for ids in chunks[start:start + self.config.batch_size]:
                    item = self.tokenizer.prepare_for_model(ids, add_special_tokens=True,
                               return_special_tokens_mask=False, truncation=False)
                    # Fast tokenizers cannot infer this mask in prepare_for_model.
                    # Structural sentinels distinguish ADDED special tokens from body
                    # [UNK]/[MASK] tokens, which must still participate in pooling.
                    layout = self.tokenizer.build_inputs_with_special_tokens([-1] * len(ids))
                    item["special_tokens_mask"] = [int(token != -1) for token in layout]
                    encoded.append(item)
                batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
                special = batch.pop("special_tokens_mask").bool().to(self.device)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                body = batch["attention_mask"].bool() & ~special
                if not body.any(dim=1).all():
                    raise ValueError("A text chunk contains no body tokens")
                hidden = self.model(**batch).last_hidden_state
                mean = hidden.masked_fill(~body[..., None], 0).sum(dim=1) / body.sum(dim=1, keepdim=True)
                vectors.append(mean.float().cpu().numpy())
        return np.concatenate(vectors)

    def export(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)


def make_backend(config, *, snapshot=None, expected_identity=None):
    if config.backend != "macbert":
        raise ValueError("Only the MacBERT text backend is supported")
    return MacBertBackend(config, snapshot=snapshot, expected_identity=expected_identity)

