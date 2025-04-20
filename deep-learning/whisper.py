import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import numpy as np
    import os
    from tokenizers import Tokenizer
    import soundfile as sf
    import warnings
    from typing import List, Optional, Union, Dict
    from datasets import load_dataset
    import numpy as np
    return (
        Dict,
        List,
        Optional,
        Tokenizer,
        Union,
        load_dataset,
        np,
        os,
        sf,
        torch,
        warnings,
    )


@app.cell
def _(Dict, List, Optional, Union, np, warnings):
    def hertz_to_mel(
        freq: Union[float, np.ndarray], mel_scale: str = "htk"
    ) -> Union[float, np.ndarray]:
        if mel_scale not in ["slaney", "htk", "kaldi"]:
            raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

        if mel_scale == "htk":
            return 2595.0 * np.log10(1.0 + (freq / 700.0))
        elif mel_scale == "kaldi":
            return 1127.0 * np.log(1.0 + (freq / 700.0))

        min_log_hertz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)
        mels = 3.0 * freq / 200.0

        if isinstance(freq, np.ndarray):
            log_region = freq >= min_log_hertz
            mels[log_region] = (
                min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
            )
        elif freq >= min_log_hertz:
            mels = min_log_mel + np.log(freq / min_log_hertz) * logstep

        return mels


    def mel_to_hertz(
        mels: Union[float, np.ndarray], mel_scale: str = "htk"
    ) -> Union[float, np.ndarray]:
        if mel_scale not in ["slaney", "htk", "kaldi"]:
            raise ValueError('mel_scale should be one of "htk", "slaney" or "kaldi".')

        if mel_scale == "htk":
            return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
        elif mel_scale == "kaldi":
            return 700.0 * (np.exp(mels / 1127.0) - 1.0)

        min_log_hertz = 1000.0
        min_log_mel = 15.0
        logstep = np.log(6.4) / 27.0
        freq = 200.0 * mels / 3.0

        if isinstance(mels, np.ndarray):
            log_region = mels >= min_log_mel
            freq[log_region] = min_log_hertz * np.exp(
                logstep * (mels[log_region] - min_log_mel)
            )
        elif mels >= min_log_mel:
            freq = min_log_hertz * np.exp(logstep * (mels - min_log_mel))

        return freq


    def _create_triangular_filter_bank(
        fft_freqs: np.ndarray, filter_freqs: np.ndarray
    ) -> np.ndarray:
        filter_diff = np.diff(filter_freqs)
        slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
        down_slopes = -slopes[:, :-2] / filter_diff[:-1]
        up_slopes = slopes[:, 2:] / filter_diff[1:]
        return np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))


    def mel_filter_bank(
        num_frequency_bins: int,
        num_mel_filters: int,
        min_frequency: float,
        max_frequency: float,
        sampling_rate: int,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        triangularize_in_mel_space: bool = False,
    ) -> np.ndarray:
        if norm is not None and norm != "slaney":
            raise ValueError('norm must be one of None or "slaney"')

        # center points of the triangular mel filters
        mel_min = hertz_to_mel(min_frequency, mel_scale=mel_scale)
        mel_max = hertz_to_mel(max_frequency, mel_scale=mel_scale)
        mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
        filter_freqs = mel_to_hertz(mel_freqs, mel_scale=mel_scale)

        if triangularize_in_mel_space:
            # frequencies of FFT bins in Hz, but filters triangularized in mel space
            fft_bin_width = sampling_rate / (num_frequency_bins * 2)
            fft_freqs = hertz_to_mel(
                fft_bin_width * np.arange(num_frequency_bins), mel_scale=mel_scale
            )
            filter_freqs = mel_freqs
        else:
            # frequencies of FFT bins in Hz
            fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

        mel_filters = _create_triangular_filter_bank(fft_freqs, filter_freqs)

        if norm is not None and norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (
                filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters]
            )
            mel_filters *= np.expand_dims(enorm, 0)

        if (mel_filters.max(axis=0) == 0.0).any():
            warnings.warn(
                "At least one mel filter has all zero values. "
                f"The value for `num_mel_filters` ({num_mel_filters}) may be set too high. "
                f"Or, the value for `num_frequency_bins` ({num_frequency_bins}) may be set too low."
            )

        return mel_filters


    def window_function(
        window_length: int,
        name: str = "hann",
        periodic: bool = True,
        frame_length: Optional[int] = None,
        center: bool = True,
    ) -> np.ndarray:
        length = window_length + 1 if periodic else window_length

        if name == "boxcar":
            window = np.ones(length)
        elif name in ["hamming", "hamming_window"]:
            window = np.hamming(length)
        elif name in ["hann", "hann_window"]:
            window = np.hanning(length)
        elif name in ["povey"]:
            window = np.power(np.hanning(length), 0.85)
        else:
            raise ValueError(f"Unknown window function '{name}'")

        if periodic:
            window = window[:-1]

        if frame_length is None:
            return window

        if window_length > frame_length:
            raise ValueError(
                f"Length of the window ({window_length}) may not be larger than frame_length ({frame_length})"
            )

        padded_window = np.zeros(frame_length)
        offset = (frame_length - window_length) // 2 if center else 0
        padded_window[offset : offset + window_length] = window
        return padded_window


    def spectrogram(
        waveform: np.ndarray,
        window: np.ndarray,
        frame_length: int,
        hop_length: int,
        fft_length: Optional[int] = None,
        power: Optional[float] = 1.0,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
        preemphasis: Optional[float] = None,
        mel_filters: Optional[np.ndarray] = None,
        mel_floor: float = 1e-10,
        log_mel: Optional[str] = None,
        reference: float = 1.0,
        min_value: float = 1e-10,
        db_range: Optional[float] = None,
        remove_dc_offset: Optional[bool] = None,
        dtype: np.dtype = np.float32,
    ) -> np.ndarray:
        window_length = len(window)

        if fft_length is None:
            fft_length = frame_length

        if frame_length > fft_length:
            raise ValueError(
                f"frame_length ({frame_length}) may not be larger than fft_length ({fft_length})"
            )

        if window_length != frame_length:
            raise ValueError(
                f"Length of the window ({window_length}) must equal frame_length ({frame_length})"
            )

        if hop_length <= 0:
            raise ValueError("hop_length must be greater than zero")

        if waveform.ndim != 1:
            raise ValueError(
                f"Input waveform must have only one dimension, shape is {waveform.shape}"
            )

        if np.iscomplexobj(waveform):
            raise ValueError("Complex-valued input waveforms are not currently supported")

        if power is None and mel_filters is not None:
            raise ValueError(
                "You have provided `mel_filters` but `power` is `None`. Mel spectrogram computation is not yet supported for complex-valued spectrogram."
                "Specify `power` to fix this issue."
            )

        # center pad the waveform
        if center:
            padding = [(int(frame_length // 2), int(frame_length // 2))]
            waveform = np.pad(waveform, padding, mode=pad_mode)

        # promote to float64, since np.fft uses float64 internally
        waveform = waveform.astype(np.float64)
        window = window.astype(np.float64)

        # split waveform into frames of frame_length size
        num_frames = int(1 + np.floor((waveform.size - frame_length) / hop_length))

        num_frequency_bins = (fft_length // 2) + 1 if onesided else fft_length
        spectrogram = np.empty((num_frames, num_frequency_bins), dtype=np.complex64)

        # rfft is faster than fft
        fft_func = np.fft.rfft if onesided else np.fft.fft
        buffer = np.zeros(fft_length)

        timestep = 0
        for frame_idx in range(num_frames):
            buffer[:frame_length] = waveform[timestep : timestep + frame_length]

            if remove_dc_offset:
                buffer[:frame_length] = buffer[:frame_length] - buffer[:frame_length].mean()

            if preemphasis is not None:
                buffer[1:frame_length] -= preemphasis * buffer[: frame_length - 1]
                buffer[0] *= 1 - preemphasis

            buffer[:frame_length] *= window

            spectrogram[frame_idx] = fft_func(buffer)
            timestep += hop_length

        # note: ** is much faster than np.power
        if power is not None:
            spectrogram = np.abs(spectrogram, dtype=np.float64) ** power

        spectrogram = spectrogram.T

        if mel_filters is not None:
            spectrogram = np.maximum(mel_floor, np.dot(mel_filters.T, spectrogram))

        if power is not None and log_mel is not None:
            if log_mel == "log":
                spectrogram = np.log(spectrogram)
            elif log_mel == "log10":
                spectrogram = np.log10(spectrogram)
            elif log_mel == "dB":
                if power == 1.0:
                    spectrogram = amplitude_to_db(
                        spectrogram, reference, min_value, db_range
                    )
                elif power == 2.0:
                    spectrogram = power_to_db(spectrogram, reference, min_value, db_range)
                else:
                    raise ValueError(
                        f"Cannot use log_mel option '{log_mel}' with power {power}"
                    )
            else:
                raise ValueError(f"Unknown log_mel option: {log_mel}")

            spectrogram = np.asarray(spectrogram, dtype)

        return spectrogram


    def power_to_db(
        spectrogram: np.ndarray,
        reference: float = 1.0,
        min_value: float = 1e-10,
        db_range: Optional[float] = None,
    ) -> np.ndarray:
        if reference <= 0.0:
            raise ValueError("reference must be greater than zero")
        if min_value <= 0.0:
            raise ValueError("min_value must be greater than zero")

        reference = max(min_value, reference)

        spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
        spectrogram = 10.0 * (np.log10(spectrogram) - np.log10(reference))

        if db_range is not None:
            if db_range <= 0.0:
                raise ValueError("db_range must be greater than zero")
            spectrogram = np.clip(
                spectrogram, a_min=spectrogram.max() - db_range, a_max=None
            )

        return spectrogram


    def amplitude_to_db(
        spectrogram: np.ndarray,
        reference: float = 1.0,
        min_value: float = 1e-5,
        db_range: Optional[float] = None,
    ) -> np.ndarray:
        if reference <= 0.0:
            raise ValueError("reference must be greater than zero")
        if min_value <= 0.0:
            raise ValueError("min_value must be greater than zero")

        reference = max(min_value, reference)

        spectrogram = np.clip(spectrogram, a_min=min_value, a_max=None)
        spectrogram = 20.0 * (np.log10(spectrogram) - np.log10(reference))

        if db_range is not None:
            if db_range <= 0.0:
                raise ValueError("db_range must be greater than zero")
            spectrogram = np.clip(
                spectrogram, a_min=spectrogram.max() - db_range, a_max=None
            )

        return spectrogram


    def extract_fbank_features(
        waveform_batch: np.array, n_fft: int, hop_length: int, mel_filters: np.ndarray
    ) -> np.ndarray:
        log_spec_batch = []
        for waveform in waveform_batch:
            log_spec = spectrogram(
                waveform,
                window_function(n_fft, "hann"),
                frame_length=n_fft,
                hop_length=hop_length,
                power=2.0,
                mel_filters=mel_filters,
                log_mel="log10",
            )
            log_spec = log_spec[:, :-1]
            log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
            log_spec = (log_spec + 4.0) / 4.0
            log_spec_batch.append(log_spec)
        log_spec_batch = np.array(log_spec_batch)
        return log_spec_batch


    def pad(
        processed_features: List[np.ndarray],
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding_side: str = "right",
        padding_value: float = 0.0,
    ) -> Dict[str, np.ndarray]:

        required_input = processed_features
        batch_size = len(required_input)

        if truncation:
            processed_features = [seq[:max_length] for seq in required_input]

        if padding:
            max_seq_len = (
                max(len(seq) for seq in required_input)
                if max_length is None
                else max_length
            )
            for i in range(batch_size):
                seq_len = len(required_input[i])
                pad_len = max_seq_len - seq_len
                if padding_side == "right":
                    padding = [(0, pad_len)] + [(0, 0)] * (required_input[i].ndim - 1)
                else:
                    padding = [(pad_len, 0)] + [(0, 0)] * (required_input[i].ndim - 1)

                processed_features[i] = np.pad(
                    required_input[i],
                    padding,
                    constant_values=padding_value,
                )
        return processed_features
    return (
        amplitude_to_db,
        extract_fbank_features,
        hertz_to_mel,
        mel_filter_bank,
        mel_to_hertz,
        pad,
        power_to_db,
        spectrogram,
        window_function,
    )


@app.cell
def _(np):
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


    def layer_norm(x, g, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return g * (x - mean) / np.sqrt(variance + eps) + b


    def linear(x, w, b):
        return x @ w + b


    def ffn(x, c_fc, c_proj):
        return linear(gelu(linear(x, **c_fc)), **c_proj)


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


    def transformer_block(x, mlp, attn, ln_1, ln_2, n_head, mask_enabled=False):
        x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head, mask_enabled=mask_enabled)
        x = x + ffn(layer_norm(x, **ln_2), **mlp)
        return x


    def decoder_transformer_block(
        x, mlp, attn, encoder_attn, ln_1, ln_2, ln_3, n_head, kv_states=None
    ):
        x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head, mask_enabled=True)
        if kv_states is not None:
            x = x + mha(
                layer_norm(x, **ln_2), **encoder_attn, kv_states=kv_states, n_head=n_head
            )
        x = x + ffn(layer_norm(x, **ln_3), **mlp)
        return x
    return (
        attention,
        decoder_transformer_block,
        ffn,
        gelu,
        layer_norm,
        linear,
        mha,
        softmax,
        transformer_block,
    )


@app.cell
def _(np):
    def convolution(input_tensor, weights, bias, stride=1, padding=0):
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
    return (convolution,)


@app.cell
def _(
    convolution,
    decoder_transformer_block,
    gelu,
    layer_norm,
    np,
    transformer_block,
):
    def whisper_encoder(audio_features, params, hparams):
        # Convolutional layers
        x = gelu(
            convolution(
                audio_features,
                params["encoder"]["conv1"]["w"],
                params["encoder"]["conv1"]["b"],
                padding=1,
            )
        )
        x = gelu(
            convolution(
                x,
                params["encoder"]["conv2"]["w"],
                params["encoder"]["conv2"]["b"],
                stride=2,
                padding=1,
            )
        )

        # Add positional embeddings
        x = x.T + params["encoder"]["embed_positions"]
        # Transformer layers
        for layer in params["encoder"]["blocks"]:
            x = transformer_block(x, **layer, n_head=hparams["n_head"])

        # Final layer norm
        x = layer_norm(x, **params["encoder"]["ln_f"])
        return x


    def whisper_decoder(encoder_output, input_ids, params, hparams):
        # Embed tokens
        token_embeddings = params["decoder"]["embed_tokens"][input_ids]
        positions = np.arange(token_embeddings.shape[0])
        token_embeddings = (
            token_embeddings + params["decoder"]["embed_positions"][positions]
        )

        # Transformer layers
        x = token_embeddings
        for layer in params["decoder"]["blocks"]:
            x = decoder_transformer_block(
                x, **layer, kv_states=encoder_output, n_head=hparams["n_head"]
            )

        # Final layer norm
        x = layer_norm(x, **params["decoder"]["ln_f"])
        return x


    def whisper_generate(audio_features, params, hparams, n_tokens):
        # Encode audio
        encoder_output = whisper_encoder(audio_features, params, hparams)

        # Initialize decoder inputs
        input_ids = [50257]  # Start of sequence token
        for _ in range(n_tokens):
            logits = whisper_decoder(encoder_output, input_ids, params, hparams)
            next_token = np.argmax(logits[-1] @ params["proj_out"]["w"])
            if next_token == 50256:
                break
            input_ids.append(next_token)
        return input_ids
    return whisper_decoder, whisper_encoder, whisper_generate


@app.cell
def _(np, torch):
    def load_whisper_parameters(
        model_path,
        encoder_layers=4,
        decoder_layers=4,
        num_attention_heads=6,
        max_position_embeddings=1024,
        hidden_dim=384,
    ):
        model = torch.load(model_path, map_location="cpu")
        # Encoder
        conv1 = {
            "w": model["model.encoder.conv1.weight"].numpy(),
            "b": model["model.encoder.conv1.bias"].numpy(),
        }
        conv2 = {
            "w": model["model.encoder.conv2.weight"].numpy(),
            "b": model["model.encoder.conv2.bias"].numpy(),
        }
        embed_positions = model["model.encoder.embed_positions.weight"].numpy()
        encoder_blocks = []
        for i in range(encoder_layers):
            q = {
                "w": model[f"model.encoder.layers.{i}.self_attn.q_proj.weight"].numpy(),
                "b": model[f"model.encoder.layers.{i}.self_attn.q_proj.bias"].numpy(),
            }
            k = {
                "w": model[f"model.encoder.layers.{i}.self_attn.k_proj.weight"].numpy(),
                "b": np.zeros((hidden_dim,)),
            }
            v = {
                "w": model[f"model.encoder.layers.{i}.self_attn.v_proj.weight"].numpy(),
                "b": model[f"model.encoder.layers.{i}.self_attn.v_proj.bias"].numpy(),
            }
            c_attn = {
                "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
                "b": np.hstack((q["b"], k["b"], v["b"])),
            }
            c_proj = {
                "w": model[f"model.encoder.layers.{i}.self_attn.out_proj.weight"].numpy().T,
                "b": model[f"model.encoder.layers.{i}.self_attn.out_proj.bias"].numpy(),
            }
            attn = {"c_attn": c_attn, "c_proj": c_proj}
            ln_1 = {
                "g": model[f"model.encoder.layers.{i}.self_attn_layer_norm.weight"].numpy(),
                "b": model[f"model.encoder.layers.{i}.self_attn_layer_norm.bias"].numpy(),
            }

            mlp_c_fc = {
                "w": model[f"model.encoder.layers.{i}.fc1.weight"].numpy().T,
                "b": model[f"model.encoder.layers.{i}.fc1.bias"].numpy(),
            }

            mlp_c_proj = {
                "w": model[f"model.encoder.layers.{i}.fc2.weight"].numpy().T,
                "b": model[f"model.encoder.layers.{i}.fc2.bias"].numpy(),
            }
            mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}

            ln_2 = {
                "g": model[f"model.encoder.layers.{i}.final_layer_norm.weight"].numpy(),
                "b": model[f"model.encoder.layers.{i}.final_layer_norm.bias"].numpy(),
            }
            block = {"mlp": mlp, "attn": attn, "ln_1": ln_1, "ln_2": ln_2}
            encoder_blocks.append(block)

        encoder_layer_norm = {
            "g": model["model.encoder.layer_norm.weight"].numpy(),
            "b": model["model.encoder.layer_norm.bias"].numpy(),
        }
        encoder = {
            "conv1": conv1,
            "conv2": conv2,
            "embed_positions": embed_positions,
            "blocks": encoder_blocks,
            "ln_f": encoder_layer_norm,
        }

        # Decoder (similar structure)
        decoder_embed_tokens = model["model.decoder.embed_tokens.weight"].numpy()
        decoder_embed_positions = model["model.decoder.embed_positions.weight"].numpy()
        decoder_blocks = []
        for i in range(decoder_layers):
            q = {
                "w": model[f"model.decoder.layers.{i}.self_attn.q_proj.weight"].numpy(),
                "b": model[f"model.decoder.layers.{i}.self_attn.q_proj.bias"].numpy(),
            }
            k = {
                "w": model[f"model.decoder.layers.{i}.self_attn.k_proj.weight"].numpy(),
                "b": np.zeros((hidden_dim,)),
            }
            v = {
                "w": model[f"model.decoder.layers.{i}.self_attn.v_proj.weight"].numpy(),
                "b": model[f"model.decoder.layers.{i}.self_attn.v_proj.bias"].numpy(),
            }
            c_attn = {
                "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
                "b": np.hstack((q["b"], k["b"], v["b"])),
            }
            c_proj = {
                "w": model[f"model.decoder.layers.{i}.self_attn.out_proj.weight"].numpy().T,
                "b": model[f"model.decoder.layers.{i}.self_attn.out_proj.bias"].numpy(),
            }
            attn = {"c_attn": c_attn, "c_proj": c_proj}
            ln_1 = {
                "g": model[f"model.decoder.layers.{i}.self_attn_layer_norm.weight"].numpy(),
                "b": model[f"model.decoder.layers.{i}.self_attn_layer_norm.bias"].numpy(),
            }

            q = {
                "w": model[f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"].numpy(),
                "b": model[f"model.decoder.layers.{i}.encoder_attn.q_proj.bias"].numpy(),
            }
            k = {
                "w": model[f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"].numpy(),
                "b": np.zeros((hidden_dim,)),
            }
            v = {
                "w": model[f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"].numpy(),
                "b": model[f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"].numpy(),
            }
            c_attn = {
                "w": np.hstack((q["w"].T, k["w"].T, v["w"].T)),
                "b": np.hstack((q["b"], k["b"], v["b"])),
            }
            c_proj = {
                "w": model[f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"]
                .numpy()
                .T,
                "b": model[f"model.decoder.layers.{i}.encoder_attn.out_proj.bias"].numpy(),
            }
            encoder_attn = {"c_attn": c_attn, "c_proj": c_proj}
            ln_2 = {
                "g": model[
                    f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"
                ].numpy(),
                "b": model[
                    f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"
                ].numpy(),
            }

            mlp_c_fc = {
                "w": model[f"model.decoder.layers.{i}.fc1.weight"].numpy().T,
                "b": model[f"model.decoder.layers.{i}.fc1.bias"].numpy(),
            }

            mlp_c_proj = {
                "w": model[f"model.decoder.layers.{i}.fc2.weight"].numpy().T,
                "b": model[f"model.decoder.layers.{i}.fc2.bias"].numpy(),
            }

            mlp = {"c_fc": mlp_c_fc, "c_proj": mlp_c_proj}

            ln_3 = {
                "g": model[f"model.decoder.layers.{i}.final_layer_norm.weight"].numpy(),
                "b": model[f"model.decoder.layers.{i}.final_layer_norm.bias"].numpy(),
            }
            block = {
                "mlp": mlp,
                "attn": attn,
                "encoder_attn": encoder_attn,
                "ln_1": ln_1,
                "ln_2": ln_2,
                "ln_3": ln_3,
            }
            decoder_blocks.append(block)
        decoder_layer_norm = {
            "g": model["model.decoder.layer_norm.weight"].numpy(),
            "b": model["model.decoder.layer_norm.bias"].numpy(),
        }
        decoder = {
            "embed_tokens": decoder_embed_tokens,
            "embed_positions": decoder_embed_positions,
            "blocks": decoder_blocks,
            "ln_f": decoder_layer_norm,
        }

        params = {
            "encoder": encoder,
            "decoder": decoder,
            "proj_out": {
                "w": model["model.decoder.embed_tokens.weight"].numpy().T,
            },
        }
        hparams = {
            "n_head": num_attention_heads,
            "n_ctx": max_position_embeddings,
        }
        return hparams, params
    return (load_whisper_parameters,)


@app.cell
def _():
    # !wget https://huggingface.co/openai/whisper-tiny.en/resolve/main/pytorch_model.bin
    # !wget https://huggingface.co/openai/whisper-tiny.en/resolve/main/tokenizer.json

    MODEL_PATH = "/path/to/pytorch_model.bin"
    TOKENIZER_PATH = "/path/to/tokenizer.json"
    return MODEL_PATH, TOKENIZER_PATH


@app.cell
def _(
    MODEL_PATH,
    TOKENIZER_PATH,
    Tokenizer,
    load_whisper_parameters,
    mel_filter_bank,
):
    hparams, params = load_whisper_parameters(MODEL_PATH)
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    feature_size = 80
    sampling_rate = 16000
    n_fft = 400
    hop_length = 160
    chunk_length = 30
    n_samples = chunk_length * sampling_rate

    mel_filters = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,
        num_mel_filters=feature_size,
        min_frequency=0.0,
        max_frequency=8000.0,
        sampling_rate=sampling_rate,
        norm="slaney",
        mel_scale="slaney",
    )
    return (
        chunk_length,
        feature_size,
        hop_length,
        hparams,
        mel_filters,
        n_fft,
        n_samples,
        params,
        sampling_rate,
        tokenizer,
    )


@app.cell
def _(load_dataset):
    # load dummy dataset and read audio files
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[10]["audio"]
    return ds, sample


@app.cell
def _(
    extract_fbank_features,
    hop_length,
    hparams,
    mel_filters,
    n_fft,
    n_samples,
    pad,
    params,
    sample,
    tokenizer,
    whisper_generate,
):
    audio_features = extract_fbank_features(
        pad([sample["array"]], max_length=n_samples),
        n_fft=n_fft,
        hop_length=hop_length,
        mel_filters=mel_filters,
    )
    ids = whisper_generate(audio_features[0], params, hparams,100)
    print(tokenizer.decode(ids))
    return audio_features, ids


if __name__ == "__main__":
    app.run()
