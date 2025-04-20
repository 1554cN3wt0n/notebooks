import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import torch
    return np, torch


@app.cell
def _(np):
    def bicubic_kernel(x, a=-0.75):
        abs_x = np.abs(x)
        abs_x2 = abs_x**2
        abs_x3 = abs_x**3

        result = np.where(
            abs_x <= 1,
            (a + 2) * abs_x3 - (a + 3) * abs_x2 + 1,
            np.where(
                abs_x <= 2,
                a * abs_x3 - 5 * a * abs_x2 + 8 * a * abs_x - 4 * a,
                0,
            ),
        )
        return result


    def bicubic_interpolate(image, x, y):
        _, h, w = image.shape

        # Get integer and fractional parts of coordinates
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        dx = x - x0
        dy = y - y0

        # Initialize result
        result = 0.0

        # Iterate over the 4x4 neighborhood
        for m in range(-1, 3):
            for n in range(-1, 3):
                xi = np.clip(x0 + m, 0, w - 1)
                yi = np.clip(y0 + n, 0, h - 1)

                weight = bicubic_kernel(m - dx) * bicubic_kernel(n - dy)
                result += weight * image[:, yi, xi]

        return result


    def resize_bicubic(image, new_width, new_height):
        n, h, w = image.shape
        resized_image = np.zeros((n, new_height, new_width))

        # Scale factors
        scale_x = w / new_width
        scale_y = h / new_height

        for j in range(new_height):
            for i in range(new_width):
                x = (i + 0.5) * scale_x - 0.5  # Map to source space
                y = (j + 0.5) * scale_y - 0.5  # Map to source space

                resized_image[:, j, i] = bicubic_interpolate(image, x, y)

        return resized_image


    def resize_image(image, new_height, new_width):
        _, old_height, old_width = image.shape

        row_scale = old_height / new_height
        col_scale = old_width / new_width

        row_indices = (np.arange(new_height) * row_scale).astype(int)
        col_indices = (np.arange(new_width) * col_scale).astype(int)

        resized_image = image[:, row_indices[:, None], col_indices]

        return resized_image

    return bicubic_interpolate, bicubic_kernel, resize_bicubic, resize_image


@app.cell
def _(np, torch):
    def load_hparams_and_params(model_path):
        n_encoder_layers = 12
        n_cls_blocks = 3
        n_bbox_blocks = 3
        hidden_dim = 192

        model = torch.load(model_path, map_location="cpu")
        cls_token = model["vit.embeddings.cls_token"].numpy().reshape(1, hidden_dim)
        detection_tokens = (
            model["vit.embeddings.detection_tokens"].numpy().reshape(-1, hidden_dim)
        )
        position_embeddings = (
            model["vit.embeddings.position_embeddings"].numpy().reshape(-1, hidden_dim)
        )

        conv_proj = {
            "w": model["vit.embeddings.patch_embeddings.projection.weight"]
            .numpy()
            .transpose(1, 0, 2, 3),
            "b": model["vit.embeddings.patch_embeddings.projection.bias"].numpy(),
        }

        embeddings = {
            "cls_token": cls_token,
            "detection_tokens": detection_tokens,
            "position_embeddings": position_embeddings,
            "conv_proj": conv_proj,
        }

        encoder_blocks = []
        for i in range(n_encoder_layers):
            q = {
                "w": model[
                    f"vit.encoder.layer.{i}.attention.attention.query.weight"
                ].numpy(),
                "b": model[f"vit.encoder.layer.{i}.attention.attention.query.bias"].numpy(),
            }
            k = {
                "w": model[f"vit.encoder.layer.{i}.attention.attention.key.weight"].numpy(),
                "b": model[f"vit.encoder.layer.{i}.attention.attention.key.bias"].numpy(),
            }
            v = {
                "w": model[
                    f"vit.encoder.layer.{i}.attention.attention.value.weight"
                ].numpy(),
                "b": model[f"vit.encoder.layer.{i}.attention.attention.value.bias"].numpy(),
            }
            c_attn = {
                "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
                "b": np.hstack((q["b"], k["b"], v["b"])),
            }
            c_proj = {
                "w": model[f"vit.encoder.layer.{i}.attention.output.dense.weight"]
                .numpy()
                .T,
                "b": model[f"vit.encoder.layer.{i}.attention.output.dense.bias"].numpy(),
            }
            attn = {"c_attn": c_attn, "c_proj": c_proj}
            ln_1 = {
                "g": model[f"vit.encoder.layer.{i}.layernorm_before.weight"].numpy(),
                "b": model[f"vit.encoder.layer.{i}.layernorm_before.bias"].numpy(),
            }

            mlp_c_fc = {
                "w": model[f"vit.encoder.layer.{i}.intermediate.dense.weight"].numpy().T,
                "b": model[f"vit.encoder.layer.{i}.intermediate.dense.bias"].numpy(),
            }

            mlp_c_proj = {
                "w": model[f"vit.encoder.layer.{i}.output.dense.weight"].numpy().T,
                "b": model[f"vit.encoder.layer.{i}.output.dense.bias"].numpy(),
            }

            mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}

            ln_2 = {
                "g": model[f"vit.encoder.layer.{i}.layernorm_after.weight"].numpy(),
                "b": model[f"vit.encoder.layer.{i}.layernorm_after.bias"].numpy(),
            }
            block = {"mlp": mlp, "attn": attn, "ln_1": ln_1, "ln_2": ln_2}
            encoder_blocks.append(block)
        ln_f = {
            "g": model["vit.layernorm.weight"].numpy().T,
            "b": model["vit.layernorm.bias"].numpy(),
        }

        clc_blocks = []
        for i in range(n_cls_blocks):
            clc_blocks.append(
                {
                    "w": model[f"class_labels_classifier.layers.{i}.weight"].numpy().T,
                    "b": model[f"class_labels_classifier.layers.{i}.bias"].numpy(),
                }
            )

        bbox_blocks = []
        for i in range(n_bbox_blocks):
            bbox_blocks.append(
                {
                    "w": model[f"bbox_predictor.layers.{i}.weight"].numpy().T,
                    "b": model[f"bbox_predictor.layers.{i}.bias"].numpy(),
                }
            )

        params = {
            "embeddings": embeddings,
            "encoder_blocks": encoder_blocks,
            "ln_f": ln_f,
            "clc_blocks": clc_blocks,
            "bbox_blocks": bbox_blocks,
        }
        hparams = {}
        hparams["n_head"] = 3
        hparams["n_ctx"] = 1024
        return hparams, params
    return (load_hparams_and_params,)


@app.cell
def _(np):
    def sample_probs(probs, temperature=1.0, top_p=0.3):
        sorted_probs = np.sort(probs)[::-1]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = sorted_probs[np.argmax(cumulative_probs > top_p)]
        probs[probs < cutoff] = 0
        probs = probs ** (1 / temperature)
        return np.random.choice(a=len(probs), p=probs / np.sum(probs))


    def mean_pooling_and_normalization(x):
        o = np.mean(x, axis=0)
        return o / np.linalg.norm(o)


    def gauss_norm(x: np.ndarray) -> np.ndarray:
        x = (x - x.mean()) / x.std()
        return x
    return gauss_norm, mean_pooling_and_normalization, sample_probs


@app.cell
def _(np):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


    def relu(x):
        return np.maximum(0, x)


    def silu(x):
        return x / (1.0 + np.exp(-x))


    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    def rms_norm(x):
        return x / np.sqrt(np.square(x).mean(-1, keepdims=True) + 1e-6)


    def layer_norm(x, g, b, eps=1e-12):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(variance + eps) + b


    def linear(x, w, b):
        return x @ w + b


    def ffn(x, c_fc, c_proj, act_fn=gelu):
        return linear(act_fn(linear(x, **c_fc)), **c_proj)


    def attention(q, k, v, mask=None):
        if mask is None:
            return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v
        return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


    def mha(x, c_attn, c_proj, n_head, kv_states=None, mask_enabled=False):
        x = linear(x, **c_attn)

        if kv_states is not None:
            qkv = []
            dim = c_attn["w"].shape[0]
            kv = linear(kv_states, **c_attn)
            qkv = [x[:, :dim], kv[:, dim : 2 * dim], kv[:, 2 * dim :]]
        else:
            qkv = np.split(x, 3, axis=-1)
        qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))
        causal_mask = None
        if mask_enabled:
            causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10
        out_heads = [attention(q, k, v, mask=causal_mask) for q, k, v in zip(*qkv_heads)]
        x = linear(np.hstack(out_heads), **c_proj)
        return x


    def convolution_1d(input_tensor, weights, bias, stride=1, padding=0):
        # Get dimensions
        in_channels, input_length = input_tensor.shape
        out_channels, _, kernel_size = weights.shape

        # Apply padding to the input tensor
        if padding > 0:
            input_tensor = np.pad(
                input_tensor,
                ((0, 0), (padding, padding)),
                mode="constant",
                constant_values=0,
            )

        # Calculate output length
        output_length = (input_length + 2 * padding - kernel_size) // stride + 1

        # Extract sliding windows (using strides)
        strided_indices = np.lib.stride_tricks.sliding_window_view(
            input_tensor, kernel_size, axis=1
        )
        # Shape of strided_indices: (in_channels, output_length, kernel_size)
        strided_indices = strided_indices[:, ::stride, :]  # Apply stride

        # Perform the convolution using broadcasting and summation
        output_tensor = np.tensordot(weights, strided_indices, axes=([1, 2], [0, 2]))
        # Shape of output_tensor: (out_channels, output_length)

        # Add bias to each output channel
        output_tensor += bias[:, None]  # Bias broadcasted to match output shape

        return output_tensor


    def convolution_2d(image, kernel, bias=None, stride=2, padding=0):
        # Extract dimensions
        in_channels, img_width, img_height = image.shape
        in_channels_k, out_channels, k_width, k_height = kernel.shape

        # Ensure the kernel matches input channels
        if in_channels != in_channels_k:
            raise ValueError(
                "The number of input channels in the image and kernel must match."
            )

        # Add padding to the image
        if padding > 0:
            image = np.pad(
                image,
                pad_width=((0, 0), (padding, padding), (padding, padding)),
                mode="constant",
                constant_values=0,
            )

        # Calculate output dimensions
        out_width = (image.shape[1] - k_width) // stride + 1
        out_height = (image.shape[2] - k_height) // stride + 1

        # Use stride tricks to create a sliding window view of the image
        shape = (in_channels, out_width, out_height, k_width, k_height)
        strides = (
            image.strides[0],
            stride * image.strides[1],
            stride * image.strides[2],
            image.strides[1],
            image.strides[2],
        )

        sliding_windows = np.lib.stride_tricks.as_strided(
            image, shape=shape, strides=strides
        )

        # Perform the convolution
        conv_result = np.einsum("cxykh,cokh->oxy", sliding_windows, kernel)

        # Add bias if provided
        if bias is not None:
            if bias.shape[0] != out_channels:
                raise ValueError("Bias shape must match the number of output channels.")
            conv_result += bias[:, None, None]

        return conv_result
    return (
        attention,
        convolution_1d,
        convolution_2d,
        ffn,
        gelu,
        layer_norm,
        linear,
        mha,
        relu,
        rms_norm,
        sigmoid,
        silu,
        softmax,
    )


@app.cell
def _(
    convolution_2d,
    ffn,
    layer_norm,
    linear,
    mha,
    np,
    relu,
    resize_bicubic,
    sigmoid,
):
    def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
        x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
        x = x + ffn(layer_norm(x, **ln_2), **mlp)
        return x


    def yolos_interpolation(
        position_embeddings,
        detection_tokens,
        img_size,
        patch_size=16,
        config_image_size=(800, 1333),
    ):
        num_detection_tokens = detection_tokens.shape[0]
        cls_pos_emb = position_embeddings[:1]
        det_pos_emb = position_embeddings[-num_detection_tokens:]

        patch_pos_emb = position_embeddings[1:-num_detection_tokens]
        patch_pos_emb = patch_pos_emb.T
        hidden_size, seq_len = patch_pos_emb.shape

        patch_height, patch_width = (
            config_image_size[0] // patch_size,
            config_image_size[1] // patch_size,
        )
        patch_pos_emb = patch_pos_emb.reshape(hidden_size, patch_height, patch_width)

        height, width = img_size
        new_patch_height, new_patch_width = (
            height // patch_size,
            width // patch_size,
        )

        patch_pos_emb = resize_bicubic(patch_pos_emb, new_patch_height, new_patch_width)

        patch_pos_emb = patch_pos_emb.reshape(hidden_size, -1).transpose(1, 0)

        scale_pos_emb = np.concatenate([cls_pos_emb, patch_pos_emb, det_pos_emb])
        return scale_pos_emb


    def yolos_embeddings(
        inputs, cls_token, detection_tokens, position_embeddings, conv_proj
    ):
        x = convolution_2d(
            inputs,
            conv_proj["w"],
            bias=conv_proj["b"],
            stride=16,
        )
        x = x.reshape(x.shape[0], -1).T
        x = np.vstack([cls_token, x, detection_tokens])

        scale_pos_emb = yolos_interpolation(
            position_embeddings,
            detection_tokens,
            img_size=(inputs.shape[1], inputs.shape[2]),
        )
        return scale_pos_emb + x


    def yolos(inputs, embeddings, encoder_blocks, ln_f, clc_blocks, bbox_blocks, n_head):
        x = yolos_embeddings(inputs, **embeddings)
        for block in encoder_blocks:
            x = transformer_block(x, **block, n_head=n_head)
        x = layer_norm(x, **ln_f)
        classes = x[-100:, :]
        bboxes = x[-100:, :]
        for i, block in enumerate(clc_blocks):
            if i == len(clc_blocks) - 1:
                classes = linear(classes, **block)
            else:
                classes = relu(linear(classes, **block))
        for i, block in enumerate(bbox_blocks):
            if i == len(bbox_blocks) - 1:
                bboxes = linear(bboxes, **block)
            else:
                bboxes = relu(linear(bboxes, **block))
        return classes, sigmoid(bboxes)
    return transformer_block, yolos, yolos_embeddings, yolos_interpolation


@app.cell
def _():
    # !wget https://huggingface.co/hustvl/yolos-tiny/resolve/main/pytorch_model.bin
    MODEL_PATH = "/path/to/pytorch_model.bin"
    return (MODEL_PATH,)


@app.cell
def _(MODEL_PATH, load_hparams_and_params):
    hparams, params = load_hparams_and_params(MODEL_PATH)
    return hparams, params


@app.cell
def _(gauss_norm, np, params, yolos):
    from PIL import Image
    import requests

    url = "https://images.unsplash.com/file-1705123271268-c3eaf6a79b21image?w=416&dpr=2&auto=format&fit=crop&q=60"
    image = Image.open(requests.get(url, stream=True).raw)


    raw_img = (
        np.array(image.getdata())
        .reshape(image.height, image.width, 3)
        .transpose(2, 0, 1)
        .astype(float)
    )
    raw_img = gauss_norm(raw_img / 255)

    classes, boxes = yolos(raw_img, **params, n_head=3)
    print(np.argmax(classes, axis=1))
    print(boxes)
    return Image, boxes, classes, image, raw_img, requests, url


if __name__ == "__main__":
    app.run()
