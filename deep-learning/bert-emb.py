import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    # !mkdir bert_emb
    # !wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin
    # !wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
    # !mv pytorch_model.bin bert_emb/
    # !mv tokenizer.json bert_emb/

    EMB_MODEL_PATH = "/path/to/pytorch_model.bin"
    EMB_TOKENIZER_PATH = "/path/to/tokenizer.json"
    return EMB_MODEL_PATH, EMB_TOKENIZER_PATH


@app.cell
def _():
    import numpy as np
    import random
    import torch
    from tokenizers import Tokenizer
    import os
    return Tokenizer, np, os, random, torch


@app.cell
def _(np, torch):
    def load_encoder_hparams_and_params(model_path, device="cpu"):
        n_layers = 6
        prefix = ""
        model = torch.load(model_path, map_location=device)
        # vocab embedding. shape [vocab_size, emb_dim] ex. [50257, 384]
        wte = model[f"{prefix}embeddings.word_embeddings.weight"].numpy()
        # context embedding. shape [ctx_len, emb_dim] ex. [1024, 384]
        wpe = model[f"{prefix}embeddings.position_embeddings.weight"].numpy()
        # Token type embedding. shape [2, emb_dim] ex. [2, 384]
        wtte = model[f"{prefix}embeddings.token_type_embeddings.weight"].numpy()

        ln_0 = {
            "g": model[f"{prefix}embeddings.LayerNorm.weight"].numpy(),
            "b": model[f"{prefix}embeddings.LayerNorm.bias"].numpy(),
        }
        blocks = []
        for i in range(n_layers):
            q = {
                "w": model[
                    f"{prefix}encoder.layer.{i}.attention.self.query.weight"
                ].numpy(),
                "b": model[f"{prefix}encoder.layer.{i}.attention.self.query.bias"].numpy(),
            }
            k = {
                "w": model[f"{prefix}encoder.layer.{i}.attention.self.key.weight"].numpy(),
                "b": model[f"{prefix}encoder.layer.{i}.attention.self.key.bias"].numpy(),
            }
            v = {
                "w": model[
                    f"{prefix}encoder.layer.{i}.attention.self.value.weight"
                ].numpy(),
                "b": model[f"{prefix}encoder.layer.{i}.attention.self.value.bias"].numpy(),
            }
            c_attn = {
                "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
                "b": np.hstack((q["b"], k["b"], v["b"])),
            }
            c_proj = {
                "w": model[f"{prefix}encoder.layer.{i}.attention.output.dense.weight"]
                .numpy()
                .T,
                "b": model[
                    f"{prefix}encoder.layer.{i}.attention.output.dense.bias"
                ].numpy(),
            }
            attn = {"c_attn": c_attn, "c_proj": c_proj}
            ln_1 = {
                "g": model[
                    f"{prefix}encoder.layer.{i}.attention.output.LayerNorm.weight"
                ].numpy(),
                "b": model[
                    f"{prefix}encoder.layer.{i}.attention.output.LayerNorm.bias"
                ].numpy(),
            }

            mlp_c_fc = {
                "w": model[f"{prefix}encoder.layer.{i}.intermediate.dense.weight"]
                .numpy()
                .T,
                "b": model[f"{prefix}encoder.layer.{i}.intermediate.dense.bias"].numpy(),
            }

            mlp_c_proj = {
                "w": model[f"{prefix}encoder.layer.{i}.output.dense.weight"].numpy().T,
                "b": model[f"{prefix}encoder.layer.{i}.output.dense.bias"].numpy(),
            }

            mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}

            ln_2 = {
                "g": model[f"{prefix}encoder.layer.{i}.output.LayerNorm.weight"].numpy(),
                "b": model[f"{prefix}encoder.layer.{i}.output.LayerNorm.bias"].numpy(),
            }
            block = {"mlp": mlp, "attn": attn, "ln_1": ln_1, "ln_2": ln_2}
            blocks.append(block)
        pooler = {
            "w": model[f"pooler.dense.weight"].numpy().T,
            "b": model[f"pooler.dense.bias"].numpy(),
        }

        params = {
            "ln_0": ln_0,
            "wte": wte,
            "wpe": wpe,
            "wtte": wtte,
            "blocks": blocks,
            "pooler": pooler,
        }
        hparams = {}
        hparams["n_head"] = 12
        hparams["n_ctx"] = 1024
        return hparams, params
    return (load_encoder_hparams_and_params,)


@app.cell
def _(np):
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


    def relu(x):
        return np.maximum(0, x)


    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    def layer_norm(x, g, b, eps=1e-12):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(variance + eps) + b


    def linear(x, w, b):
        return x @ w + b


    def ffn(x, c_fc, c_proj,act_fun=relu):
        return linear(act_fun(linear(x, **c_fc)), **c_proj)


    def attention(q, k, v):
        return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v


    def mha(x, c_attn, c_proj, n_head):
        x = linear(x, **c_attn)
        qkv_heads = list(
            map(lambda x: np.split(x, n_head, axis=-1), np.split(x, 3, axis=-1))
        )
        out_heads = [attention(q, k, v) for q, k, v in zip(*qkv_heads)]
        x = linear(np.hstack(out_heads), **c_proj)
        return x
    return attention, ffn, gelu, layer_norm, linear, mha, relu, softmax


@app.cell
def _(np):
    def mean_pooling_and_normalization(x):
        o = np.mean(x, axis=0)
        return o / np.linalg.norm(o)
    return (mean_pooling_and_normalization,)


@app.cell
def _(ffn, gelu, layer_norm, mha):
    def bert_emb_transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
        x = layer_norm(x + mha(x, **attn, n_head=n_head), **ln_1)
        x = layer_norm(x + ffn(x, **mlp, act_fun=gelu), **ln_2)
        return x


    def bert_emb(inputs, segment_ids, wte, wpe, wtte, ln_0, blocks, pooler, n_head):
        x = wte[inputs] + wpe[range(len(inputs))] + wtte[segment_ids]
        x = layer_norm(x, **ln_0)
        for block in blocks:
            x = bert_emb_transformer_block(x, **block, n_head=n_head)
        return x
    return bert_emb, bert_emb_transformer_block


@app.cell
def _(
    EMB_MODEL_PATH,
    EMB_TOKENIZER_PATH,
    Tokenizer,
    bert_emb,
    load_encoder_hparams_and_params,
    mean_pooling_and_normalization,
    np,
):
    hparams, params = load_encoder_hparams_and_params(
        model_path=EMB_MODEL_PATH
    )

    tokenizer = Tokenizer.from_file(EMB_TOKENIZER_PATH)
    tokenizer.no_padding()

    sentences = [
        "The sun is shining brightly in the sky.",
        "It’s a clear day with plenty of sunshine.",
        "I forgot to bring my umbrella, and now it’s raining heavily.",
        "The cat is sleeping peacefully on the couch.",
    ]

    embeddings = []
    for sentence in sentences:
        sentence_ids = tokenizer.encode(sentence).ids

        logits = bert_emb(
            sentence_ids, [0] * len(sentence_ids), **params, n_head=hparams["n_head"]
        )
        embeddings.append(mean_pooling_and_normalization(logits))

    embeddings = np.vstack(embeddings)
    print(embeddings @ embeddings.T)
    return (
        embeddings,
        hparams,
        logits,
        params,
        sentence,
        sentence_ids,
        sentences,
        tokenizer,
    )


if __name__ == "__main__":
    app.run()
