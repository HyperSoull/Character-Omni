# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
import librosa
import torch.nn.functional as F
import argparse
import copy
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, Union
import logging
from huggingface_hub import hf_hub_download
import numpy as np
import sphn
import torch
from torch.profiler import profile, ProfilerActivity
import torch.nn as nn
from safetensors.torch import load_file, load_model
import whisper
from Asimov2.modeling_asimovQS3_causal5_moshi import Asimov
# Replacing 'sphn' with 'soundfile' and 'librosa'
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("--snac-weight", type=str, default="/data/jcxy/haolu/workspace/frameworks/moshi/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors")
parser.add_argument("--model", type=str, default="/data/jcxy/haolu/workspace/frameworks/moshi/output/moshi/model_01/Qwen2-0.5B-Instruct")
parser.add_argument("--mapper_model", type=str, default="/data/jcxy/haolu/workspace/frameworks/moshi/output/Asimov/model4_02/checkpoint-13000/model.safetensors")
parser.add_argument("--device", type=str, default='cuda' if torch.cuda.device_count() else 'cpu')
parser.add_argument("--profile", action='store_true')
args = parser.parse_args()


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_all(42424242)

# SAMPLE_RATE = 32000
# FRAME_RATE = 12.5
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ModelArgs:
    snac_model_path = "/data/jcxy/haolu/workspace/frameworks/moshi/SNAC-Vocos/mscodec_checkpoint_epoch=18_step=209252_val_loss=7.5564.ckpt"
    snac_config_path = "/data/jcxy/haolu/workspace/frameworks/SNAC-Vocos/config/snac_vocos_24k.yaml"
    model_name_or_path = "/data/jcxy/haolu/workspace/store/model/Qwen2-Audio-7B-Instruct"
    extra_path = "/data/jcxy/haolu/workspace/frameworks/moshi/output/Asimov/model5_qwen_test_stage2_snac34w3/checkpoint-8000/model.safetensors"

model_args = ModelArgs()
asimov = Asimov(model_args, device=args.device)

asimov.load_weights(model_args.extra_path)
history = []
def snac_streaming_test(asimov_model: Asimov, max_duration_sec=10.0):
    snac = asimov_model.snac_model  # Define 'snac' as the MimiModel instance
    sample_rate = snac.sample_rate
    asimov.processor.tokenizer.bos_token = "<|im_start|>"
    #audio_path = "/data/jcxy/hhy/workspace/aiv/GPT-SoVITS/vo_BZLQ001_6_hutao_02.wav"#{asimov.processor.tokenizer.bos_token}system\nYou are a helpful assistant.{asimov.processor.tokenizer.eos_token}\n
    audio_path = "/data/jcxy/haolu/workspace/frameworks/12345/GPT-SoVITS/audioV2/audio/conversation_9318_user_22.wav"
    text = f"{asimov.processor.tokenizer.bos_token}system\nYou are a helpful assistant.{asimov.processor.tokenizer.eos_token}\n{asimov.processor.tokenizer.bos_token}user\n<|audio_bos|><|AUDIO|><|audio_eos|>{asimov.processor.tokenizer.eos_token}\n{asimov.processor.tokenizer.bos_token}assistant\n"#这就是我们想要的。记得每一次的小进步都是值得庆祝的。你已经在走出孤独的迷雾了。
    #text = f"{asimov.processor.tokenizer.bos_token}user\n<|audio_bos|><|AUDIO|><|audio_eos|>{asimov.processor.tokenizer.eos_token}\n{asimov.processor.tokenizer.bos_token}assistant\n"
    inputs = asimov.processor(text=text, audios=librosa.load(audio_path, sr=16000)[0], sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}
    text_input_ids = asimov_model.processor.tokenizer.encode(text, return_tensors='pt').to(device=args.device)  # Shape: [1, text_seq_len]
    print("Encoded Text:", text_input_ids)
    stream = False
    # Generate new audio and text tokens using Asimov's generate_loop method
    print("Generating new audio and text tokens...")
    generator = asimov_model.generate_loop(
        full_inputs=inputs,
        max_length=600,  # Adjust as needed
        stream=stream,
    )
    all_pcms = []
    if stream:
        print("Streaming decoding of generated audio tokens...")
        # 迭代生成器，逐步接收生成的 tokens
        for step, step_output in enumerate(generator):
            generated_text_tokens = step_output["text_tokens"]  # Shape: [batch_size, gen_seq_len]
            decoded_text = asimov_model.processor.tokenizer.decode(generated_text_tokens[0], skip_special_tokens=False)
            print(f"Step {step}:Generated Text: {decoded_text}")
            generated_audio_tokens = step_output["audio_tokens"]  # Shape: [batch_size, quantization, gen_seq_len]
            print(f"Step {step}:Generated audio tokens: {generated_audio_tokens}")
            new_all_pcms = []
            if step == 0:
                all_pcms = generated_audio_tokens
            else:
                for prev_tensor,now_tensor in zip(all_pcms,generated_audio_tokens):
                    print(prev_tensor,now_tensor)
                    prev_tensor = torch.cat((prev_tensor,now_tensor),dim=-1)
                    new_all_pcms.append(prev_tensor)
                all_pcms = new_all_pcms
            # features = snac.codes_to_features(generated_audio_tokens)
            # bandwidth_id = torch.tensor([3]).to(args.device)
            # decoded_waveform = snac.decode(features,bandwidth_id)
            # #print(f"\nPCM concatenated shape: {decoded_waveform.shape}, dtype: {decoded_waveform.dtype}")
            # decoded_waveform = decoded_waveform.squeeze(0).squeeze(0)  # [T]
            # # 将张量移回 CPU 并转换为 NumPy 数组
            # decoded_waveform_np = decoded_waveform.cpu().numpy()
            # # Save the decoded audio
            # output_path_streaming = f"streaming_out_{step}.wav"
            # sf.write(output_path_streaming, decoded_waveform_np, sample_rate)
            # print(f"Saved streaming output to {output_path_streaming}")

        print("Generation completed.")
        #print(all_pcms)
        features = snac.codes_to_features(all_pcms)
        bandwidth_id = torch.tensor([3]).to(args.device)
        decoded_waveform = snac.decode(features,bandwidth_id)
        print(f"\nPCM concatenated shape: {decoded_waveform.shape}, dtype: {decoded_waveform.dtype}")
        decoded_waveform = decoded_waveform.squeeze(0).squeeze(0)  # [T]
        # 将张量移回 CPU 并转换为 NumPy 数组
        decoded_waveform_np = decoded_waveform.cpu().numpy()
        # Save the decoded audio
        output_path_streaming = "streaming_out.wav"
        sf.write(output_path_streaming, decoded_waveform_np, sample_rate)
        print(f"Saved streaming output to {output_path_streaming}")
    else:
        print("Generation completed.")
        # 收集生成的结果
        generated_tokens = None
        for item in generator:
            generated_tokens = item
        # Process generated text tokens
        generated_text_tokens = generated_tokens["text_tokens"]  # Shape: [batch_size, gen_seq_len]
        print(generated_text_tokens)
        decoded_text = asimov_model.processor.tokenizer.decode(generated_text_tokens[0], skip_special_tokens=False)
        print("Generated Text:")
        print(decoded_text)
        # Process generated audio tokens
        generated_audio_tokens = generated_tokens["audio_tokens"]  # Shape: [batch_size, quantization, gen_seq_len]
        print(generated_audio_tokens)
        #time.sleep(1e6)
        print("Streaming decoding of generated audio tokens...")
        features = snac.codes_to_features(generated_audio_tokens)
        bandwidth_id = torch.tensor([3]).to(args.device)
        
        decoded_waveform = snac.decode(features,bandwidth_id)
        print(f"\nPCM concatenated shape: {decoded_waveform.shape}, dtype: {decoded_waveform.dtype}")
        decoded_waveform = decoded_waveform.squeeze(0).squeeze(0)  # [T]
        # 将张量移回 CPU 并转换为 NumPy 数组
        decoded_waveform_np = decoded_waveform.cpu().numpy()
        # Save the decoded audio
        output_path_streaming = "streaming_out.wav"
        sf.write(output_path_streaming, decoded_waveform_np, sample_rate)
        print(f"Saved streaming output to {output_path_streaming}")

with torch.no_grad():
    snac_streaming_test(asimov)
