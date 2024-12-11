#modeling_asimov.py
import copy
from functools import partial
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, Optional, OrderedDict, Union


from encoder import quantization
from qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessor
from qwen2.modeling_qwen2 import Cache, Qwen2DecoderLayer, Qwen2ForCausalLM, Qwen2MLP, StaticCache
from qwen2.tokenization_qwen2 import Qwen2Tokenizer
from qwen2.configuration_qwen2 import Qwen2Config
import torch
from tqdm import tqdm
from safetensors.torch import load_model,load_file
import torch.nn as nn
from qwen2_audio.modeling_qwen2_audio import Qwen2AudioEncoder, Qwen2AudioForConditionalGeneration, Qwen2AudioMultiModalProjector
from qwen2_audio.configuration_qwen2_audio import Qwen2AudioConfig, Qwen2AudioEncoderConfig
import torch.nn.functional as F
import typing as tp
import torchaudio
import yaml
import math

from encoder.utils import convert_audio
from decoder.pretrained import SnacVocos

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")
# Model Constants
IGNORE_INDEX = -100
SAMPLE_RATE = 32000
FRAME_RATE = 12.5

class SnacInfer:
    def __init__(self, config_path, model_path, device):
        self.config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
        self.model = SnacVocos.from_pretrained(config_path, model_path)
        self.model = self.model.to(device)
        self.device = device
        self.hop_length = self.config["model"]["init_args"]["head"]["init_args"]["hop_length"]
        self.vq_scales = self.config["model"]["init_args"]["feature_extractor"]["init_args"]["vq_scales"]
        self.sample_rate = self.config["model"]["init_args"]["sample_rate"]
    def preprocess(self, wav):
        length = wav.shape[-1]
        pad_to = self.hop_length * self.vq_scales[0]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        wav = torch.nn.functional.pad(wav, (0, right_pad))
        return wav

    def encode_infer(self, wav, bandwidth_id):
        wav = self.preprocess(wav)
        wav = wav.to(self.device)
        features, discrete_code = self.model.encode_infer(wav, bandwidth_id=bandwidth_id)
        return features, discrete_code

    def codes_to_features(self, codes: tp.List[int]) -> torch.Tensor:
        features = self.model.feature_extractor.quantizer.decode(codes)
        return features

    def decode(self, features, bandwidth_id):
        bandwidth_id.to(self.device)
        audio_out = self.model.decode(features, bandwidth_id=bandwidth_id)
        return audio_out

    def run(self, wav_path, target_sr):
        wav, sr = torchaudio.load(wav_path)
        wav = convert_audio(wav, sr, target_sr, 1)
        wav = self.preprocess(wav)
        bandwidth_id = torch.tensor([3]).to(self.device)
        wav = wav.to(self.device)
        features, discrete_code = self.encode_infer(wav, bandwidth_id=bandwidth_id)
        audio_out = self.decode(features, bandwidth_id=bandwidth_id)

        return audio_out
def get_snac(
    config_path: str,
    model_path: str,
    device: torch.device | str = "cpu",
) -> SnacInfer:
    #config_path = "/data/jcxy/haolu/workspace/frameworks/SNAC-Vocos/trained_models/snac_vocos/mscodec/snac_vocos_nq4_scale8421_16khz/lightning_logs/version_8/config.yaml"
    #model_path = "/work/jcxy/llm_model/qwen/SNAC-Vocos/trained_models/snac_vocos_nq4_scale8421_16khz/lightning_logs/version_5/checkpoints/mscodec_checkpoint_epoch=19_step=113000_val_loss=7.5217.ckpt"
    model = SnacInfer(config_path, model_path, device)
    return model
def get_llm(filename: Optional[str | Path] = None, llm_config: dict = None, device: torch.device | str = "cpu"):
    """
    Initialize a new Language Model (LLM).

    Args:
        filename (Optional[str | Path]): Path to the pretrained model.
        device (torch.device | str): Device to load the model on.

    Returns:
        tuple: The LLM model and tokenizer.
    """
    config = Qwen2Config.from_pretrained(filename)
    config.output_hidden_states = True  # Ensure hidden states are returned
    model = Qwen2ForCausalLM.from_pretrained(filename, config=config).to(device)
    #model = AutoModelForCausalLM.from_pretrained(filename).to(device)
    tokenizer = Qwen2Tokenizer.from_pretrained(filename)

    print("loaded LLM model.")
    return model, tokenizer


from peft import LoraConfig, get_peft_model
def get_qwenaudio(model_name_or_path: str = "/data/jcxy/haolu/workspace/store/model/Qwen2-Audio-7B-Instruct", device: torch.device = torch.device("cpu")):
    from qwen2_audio.modeling_qwen2_audio import Qwen2AudioForConditionalGeneration
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_name_or_path)
    #model.output_hidden_states = True 
    model.to(device)

    return model,processor


class Vocader(nn.Module):
    def __init__(self, config, num_layers=8):
        """
        Initializes the Vocader with multiple decoder layers.

        Args:
            config: Configuration for Qwen2DecoderLayer.
            llm_model_config_hidden_size (int): Hidden size of the LLM model.
            card (int): Parameter for the Linear layer output size.
            num_layers (int, optional): Number of decoder layers. Default is 8.
        """
        super(Vocader, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
                Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(num_layers)
        ])

    def forward(self, x, attention_mask=None, position_ids=None):
        """
        Forward pass through all decoder layers.

        Args:
            x (Tensor): Input tensor to the vocader layers.

        Returns:
            Tensor: Stacked outputs from all decoder layers.
                    Shape: (batch_size, num_layers, ...)
        """
        for layer in self.layers:
            x = layer(x,attention_mask=attention_mask,position_ids=position_ids)[0]

        return x  # Shape: (batch, num_layers, ...)


from transformers.modeling_utils import PreTrainedModel
from types import SimpleNamespace

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d
    
    
class ReducedDuplicateLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction='mean', base_weight=1.0, decay_factor=0.7, card=1027, min_weight=0.1, special_token_ids=None):
        """
        自定义损失函数，减少连续重复token的影响，并引入衰减因子。

        参数:
            ignore_index (int): 被忽略的目标值，不会对损失产生贡献。
            reduction (str): 'none' | 'mean' | 'sum'，决定如何聚合损失。
            base_weight (float): 初始损失权重。
            decay_factor (float): 衰减因子，用于连续重复token的权重递减。
            card (int): 类别数。
            special_token_ids (list or set): 特殊token的ID，权重始终为1.0。
        """
        super(ReducedDuplicateLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.base_weight = base_weight
        self.decay_factor = decay_factor
        self.min_weight = min_weight  # 最小权重
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.card = card
        if special_token_ids is None:
            self.special_token_ids = set()
        else:
            self.special_token_ids = set(special_token_ids)
    def forward(self, logits, labels):
        # 计算标准交叉熵损失
        loss = self.cross_entropy(logits[..., :-1, :].contiguous().view(-1, self.card),
                                 labels[..., 1:].contiguous().view(-1))

        # 获取batch size和序列长度
        batch_size, seq_len = labels.size(0), labels.size(1)

        # 计算前一个token和当前token
        prev_labels = labels[:, :-1].contiguous().view(batch_size, -1)
        current_labels = labels[:, 1:].contiguous().view(batch_size, -1)

        # 创建mask，标识当前token是否与前一个token相同
        duplicate_mask = (current_labels == prev_labels).float()

        # 初始化重复计数器
        # 这里我们使用累积的方式来计算每个重复token的重复次数
        # 每当一个token与前一个token相同，计数器加1；否则重置为1
        # 这里的计数从1开始表示当前token本身
        with torch.no_grad():
            # 初始化计数器为1
            repeat_counts = torch.ones_like(duplicate_mask)

            # 遍历序列长度
            for i in range(1, duplicate_mask.size(1)):
                # 如果当前token是重复的，计数器加1
                # 否则重置为1
                repeat_counts[:, i] = duplicate_mask[:, i] * (repeat_counts[:, i-1] + 1) + (1 - duplicate_mask[:, i]) * 1

        # 计算衰减权重
        # 权重 = base_weight * decay_factor^(count - 1)
        weights = self.base_weight * (self.decay_factor ** (repeat_counts - 1))
        # 对于非重复token，权重为base_weight
        weights = weights * duplicate_mask + self.base_weight * (1 - duplicate_mask)
        weights = torch.clamp(weights, min=self.min_weight)
        # 设置特殊token的权重为1.0
        if self.special_token_ids:
            #print("true")
            # 将特殊tokenID转换为tensor，并移动到相同设备
            special_tokens_tensor = torch.tensor(list(self.special_token_ids), device=labels.device).unsqueeze(0)  # [1, num_special]
            # 扩展current_labels以便进行比较
            current_labels_expanded = current_labels.unsqueeze(-1)  # [batch_size, seq_len, 1]
            # 比较每个token是否是特殊token
            is_special = (current_labels_expanded == special_tokens_tensor).any(-1).float()  # [batch_size, seq_len]
            # 更新权重：特殊token权重为base_weight
            weights = weights * (1 - is_special) + self.base_weight*2.0 * is_special
        # 调整损失权重
        loss = loss.view(batch_size, -1) * weights
        #print(f"Weight stats - min: {weights.min().item()}, max: {weights.max().item()}, mean: {weights.mean().item()}")
        # 根据reduction参数聚合损失
        if self.reduction == 'mean':
            mask = (current_labels != self.ignore_index).float()
            loss = (loss * mask).sum() / mask.sum()
        elif self.reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss.view(batch_size, -1)

        return loss
    
class Asimov(nn.Module):
    def __init__(
        self,
        model_args: 'ModelArguments',
        device: Union[torch.device, str] = "cpu",
    ):
        super(Asimov, self).__init__()
        #nn.Module.__init__(self)
        self._device = device
        with open(model_args.model_name_or_path+"/config.json","r",encoding="utf-8") as f:
            model_config = json.load(f)
        model_config["model_type"] = "asimov"
        model_config["torch_dtype"] = "bfloat16"
        self.model_config = model_config
        self.card = 1024 + 3
        self.snac_head_num = 7
        self.snac_bos_token_id = 1024
        self.snac_eos_token_id = 1025
        self.snac_pad_token_id = 1026
        # Initialize Mimi model 
        self.snac_model = get_snac(model_path=model_args.snac_model_path,config_path=model_args.snac_config_path, device=device)
        for param in self.snac_model.model.parameters():
            param.requires_grad = False  # Freeze Mimi model

        self.model,self.processor = get_qwenaudio(device=device)
        for param in self.model.parameters():
            param.requires_grad = False  # Freeze
        self.model = self.model.to(device).to(dtype=torch.bfloat16)
        
        self.config_kwargs = {
            "architectures": [
                "Qwen2ForCausalLM"
            ],
            "pad_token_id": 151643,
            "bos_token_id": 151644,
            "eos_token_id": 151645,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "hidden_size": self.model.language_model.config.hidden_size,  # Reduced from 896 to 512
            "initializer_range": 0.02,
            "intermediate_size": self.model.language_model.config.intermediate_size,  # Reduced from 4864 to 2048
            "max_position_embeddings": self.model.language_model.config.max_position_embeddings,
            "max_window_layers": self.model.language_model.config.max_window_layers,  # Reduced from 24 to 2
            "num_attention_heads": self.model.language_model.config.num_attention_heads,  # Reduced from 14 to 8
            "num_key_value_heads": self.model.language_model.config.num_key_value_heads,
            "rms_norm_eps": 1e-06,
            "torch_dtype": "bfloat16",
            "attention_bias": False,
            "use_cache": True
        }

        # Initialize configuration
        config = Qwen2Config(**self.config_kwargs)
        config.output_hidden_states = True  # Ensure hidden states are returned

        config._attn_implementation = "flash_attention_2"
        self.audio_input_embedding_layers = nn.ModuleList(
            [nn.Embedding(self.card, self.model.language_model.config.hidden_size,padding_idx=self.snac_pad_token_id) for _ in range(self.snac_head_num)]
        ).to(device).to(dtype=torch.bfloat16)
        
        self.loss_fct_audio = ReducedDuplicateLoss(
            ignore_index=IGNORE_INDEX,  # 确保与标签的ignore index一致
            reduction='mean',   # 可以是 'mean', 'sum' 或 'none'
            card = self.card,
            special_token_ids=[self.snac_bos_token_id, self.snac_eos_token_id]  # 特殊token ID
        )
 
        config.hidden_size = 1024
        config.num_attention_heads = 16
        config.num_key_value_heads = 4
        config.intermediate_size = int(1024*4.125)
        self.vocader_layers = Vocader(config, 6).to(device).to(dtype=torch.bfloat16)
        self.audio_input_embedding_layers2 = nn.ModuleList(
            [nn.Embedding(self.card, config.hidden_size,padding_idx=self.snac_pad_token_id) for _ in range(self.snac_head_num)]
        ).to(device).to(dtype=torch.bfloat16)
        self.ar_lmhead = nn.Linear(config.hidden_size, self.card).to(device).to(dtype=torch.bfloat16)
        self.depformer_layers = Vocader(config, 6).to(device).to(dtype=torch.bfloat16)
        self.proj_layers = nn.Linear(self.model.language_model.config.hidden_size, config.hidden_size).to(device).to(dtype=torch.bfloat16)
        #self._copy_vocader_weights()
        for param in self.vocader_layers.parameters():
            param.requires_grad = True # Freeze
        self.nar_lmhead = nn.Linear(config.hidden_size, self.card).to(device).to(dtype=torch.bfloat16)

    def _copy_vocader_weights(self):
        """
        Copies the weights from the last four layers of the main model to the Vocader's layers.
        """
        main_model_layers = self.model.language_model.model.layers
        vocader_layers = self.vocader_layers.layers

        # Ensure that the main model has at least four layers
        if len(main_model_layers) < 4:
            raise ValueError("The main model has fewer than four layers.")

        # Get the last four layers from the main model
        source_layers = main_model_layers[-4:]

        # Ensure that Vocader has exactly four layers
        if len(vocader_layers) != 4:
            raise ValueError("Vocader should have exactly four layers.")

        # Copy the weights
        for vocader_layer, source_layer in zip(vocader_layers, source_layers):
            vocader_layer.load_state_dict(source_layer.state_dict())
            logger.info(f"Copied weights from main model layer to Vocader layer.")
    def load_weights(self, filename: str | Path) -> None:
        """Load weights from a checkpoint file, skipping parameters with size mismatches."""
        # 加载检查点的 state_dict
        state_dict = load_file(filename)  # 确保 load_file 返回的是 state_dict
        # 获取当前模型的 state_dict
        model_state_dict = self.state_dict()
        # 创建一个新的 OrderedDict 来存储匹配的参数
        filtered_state_dict = OrderedDict()
        mismatched_keys = []
        skipped_keys = []

        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].size() == value.size():
                    filtered_state_dict[key] = value
                else:
                    mismatched_keys.append(key)
                    logger.warning(
                        f"Skipping loading parameter '{key}' due to size mismatch: "
                        f"checkpoint param shape {value.size()}, model param shape {model_state_dict[key].size()}."
                    )
            else:
                skipped_keys.append(key)
                logger.warning(f"Skipping loading unexpected parameter '{key}' not found in the model.")

        # 加载过滤后的 state_dict
        load_result = self.load_state_dict(filtered_state_dict, strict=False)

        # 记录缺失的参数
        if load_result.missing_keys:
            logger.warning(f"Missing keys when loading state_dict: {load_result.missing_keys}")

        # 记录意外的参数（已经在上面循环中记录）
        if skipped_keys:
            logger.warning(f"Skipped {len(skipped_keys)} unexpected keys not found in the model.")
        if mismatched_keys:
            logger.warning(f"Skipped {len(mismatched_keys)} parameters due to size mismatch.")

        logger.info(f"Loaded weights from: {filename} with strict=False. "
                    f"{len(filtered_state_dict)} parameters successfully loaded.")



    def load_weights_skip(self, filename: Union[str, Path], skip_prefixes: Optional[list] = None) -> None:
        """Load weights from a checkpoint file, skipping parameters with size mismatches and specified prefixes."""
        if skip_prefixes is None:
            skip_prefixes = []
        
        # 加载检查点的 state_dict
        state_dict = load_file(filename)  # 确保 load_file 返回的是 state_dict
        
        # 获取当前模型的 state_dict
        model_state_dict = self.state_dict()
        
        # 创建一个新的 OrderedDict 来存储匹配的参数
        filtered_state_dict = OrderedDict()
        mismatched_keys = []
        skipped_keys = []

        for key, value in state_dict.items():
            # 检查是否需要跳过特定前缀的键
            skip = False
            for prefix in skip_prefixes:
                if key.startswith(prefix):
                    skipped_keys.append(key)
                    logger.warning(f"Skipping loading parameter '{key}' due to skip prefix '{prefix}'.")
                    skip = True
                    break
            if skip:
                continue
            
            if key in model_state_dict:
                if model_state_dict[key].size() == value.size():
                    filtered_state_dict[key] = value
                else:
                    mismatched_keys.append(key)
                    logger.warning(
                        f"Skipping loading parameter '{key}' due to size mismatch: "
                        f"checkpoint param shape {value.size()}, model param shape {model_state_dict[key].size()}."
                    )
            else:
                skipped_keys.append(key)
                logger.warning(f"Skipping loading unexpected parameter '{key}' not found in the model.")

        # 加载过滤后的 state_dict
        load_result = self.load_state_dict(filtered_state_dict, strict=False)

        # 记录缺失的参数
        if load_result.missing_keys:
            logger.warning(f"Missing keys when loading state_dict: {load_result.missing_keys}")

        # 记录意外的参数（已经在上面循环中记录）
        if skipped_keys:
            logger.warning(f"Skipped {len(skipped_keys)} unexpected keys not found or skipped due to prefixes.")
        if mismatched_keys:
            logger.warning(f"Skipped {len(mismatched_keys)} parameters due to size mismatch.")

        logger.info(f"Loaded weights from: {filename} with strict=False. "
                    f"{len(filtered_state_dict)} parameters successfully loaded.")


    def shift_really_value(self, x, audio_label):
        #print("x ", x.size())
        final_x = torch.full_like(x, self.snac_pad_token_id, dtype=torch.long, device=x.device)
        batch_size, _, seq_len = audio_label.shape

        # 计算每个样本的有效长度
        valid_lengths = torch.sum(audio_label[:, 0, :] != IGNORE_INDEX, dim=1) - 1  # 移除最后一个有效索引

        for b in range(batch_size):
            valid_length = valid_lengths[b].item()
            final_x[b, :valid_length, :] = x[b, :valid_length, :]

        x = final_x[:,:-1,:]
        
        #print("x pre", x.size())

        return x
    
    def shift_really_ids(self, x, audio_label):
        #print("x ", x.size())
        final_x = torch.full_like(x, self.snac_pad_token_id, dtype=torch.long, device=x.device)
        batch_size, _, seq_len = audio_label.shape

        # 计算每个样本的有效长度
        valid_lengths = torch.sum(audio_label[:, 0, :] != IGNORE_INDEX, dim=1) - 1  # 移除最后一个有效索引

        for b in range(batch_size):
            valid_length = valid_lengths[b].item()
            final_x[b, :valid_length] = x[b, :valid_length]

        x = final_x[:,:-1,:]
        
        #print("x pre", x.size())

        return x
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_labels: Optional[torch.Tensor] = None,
        audio_input_ids: Optional[torch.Tensor] = None,
        audio_labels: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Dict[str, Any]:
        if audio_labels is not None:
            batch_size, quantization, seq_len = audio_labels.size()
        else:
            batch_size = text_input_ids.size(0)
            quantization = 7
        outputs = None
              #################################
        outputs = self.model.forward(input_ids=text_input_ids,
                           input_features=input_features,
                           attention_mask=attention_mask,
                           feature_attention_mask=feature_attention_mask,
                           labels=text_labels,
                           use_loss=None,
                           output_hidden_states=True
                           )
        
        # Step 2: Get hidden states from LLM
        hidden_states = outputs.hidden_states[-1].to(dtype=torch.bfloat16)  # Shape: [batch_size, seq_len, hidden_size]
        text_labels = outputs.labels
        logits_text = outputs.logits
        tgt_label_reps = []
        if text_labels is not None:
            for i in range(batch_size):
                tgt_rep = hidden_states[i]          # Shape: [combined_seq_len_i, hidden_size]
                label = text_labels[i,:]             # Shape: [text_seq_len_i]
                tgt_label_reps.append(tgt_rep[label != IGNORE_INDEX])
        hidden_states = torch.nn.utils.rnn.pad_sequence(tgt_label_reps, batch_first=True,padding_value=0)
        #   #################################
        #final_text_input_ids = torch.full_like(text_input_ids, self.processor.tokenizer.pad_token_id, dtype=torch.long, device=audio_input_ids.device)
        final_audio_input_ids = torch.full_like(audio_input_ids, self.snac_pad_token_id, dtype=torch.long, device=audio_input_ids.device)
        final_audio_labels = torch.full_like(audio_labels, -100, dtype=torch.long, device=audio_input_ids.device)
        final_attention_mask = torch.full_like(attention_mask, 0, dtype=torch.long, device=attention_mask.device)
        for b in range(text_input_ids.shape[0]):
            l_idx = torch.where(text_input_ids[b] != 151646)[0]
            #final_text_input_ids[b, :l_idx.shape[0]] = text_input_ids[b, l_idx]
            final_audio_input_ids[b, :, :l_idx.shape[0]] = audio_input_ids[b, :, l_idx]
            final_audio_labels[b, :, :l_idx.shape[0]] = audio_labels[b, :, l_idx]
            final_attention_mask[b, :l_idx.shape[0]] = attention_mask[b, l_idx]

        audio_input_ids = final_audio_input_ids[:,:,:hidden_states.size(1)]
        audio_labels = final_audio_labels[:,:,:hidden_states.size(1)]
        attention_mask = final_attention_mask[:,:hidden_states.size(1)]
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)
        #logits_text = self.model.language_model.lm_head(hidden_states)
        
        # audio_input_embeds  = torch.zeros(
        #     batch_size, audio_input_ids[:,0,:].size(1), self.model.language_model.config.hidden_size,
        #     device=hidden_states.device, dtype=torch.bfloat16 
        # )
        # for i in range(quantization):
        #     audio_input_embeds += self.audio_input_embedding_layers[i](audio_input_ids[:,i,:])
        # audio_input_embeds = audio_input_embeds / self.snac_head_num
        audio_input_embeds = self.audio_input_embedding_layers[0](audio_input_ids[:,0,:])
        hidden_states = (audio_input_embeds + hidden_states)#  Element-wise addition
        batch_size = hidden_states.size(0)
        hidden_states = self.proj_layers(hidden_states)
        hidden_states = self.vocader_layers(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        if hidden_states.dim() != 3:
            hidden_states = hidden_states.unsqueeze(0)
        ar_logits = self.ar_lmhead(hidden_states)
        
        # 确保 hidden_states.size(2) 是 quantization 的倍数
        if training:#得到的结论是不用shift，因为不论怎样预测的1-10的那个10一定是特殊token
            zero_hidden_states = hidden_states
            i = random.randint(1, 6) #i代表我期望得到的通道 比如由0得到1 i=1
            i_tensor = torch.full((batch_size, 1), i, dtype=zero_hidden_states.dtype, device=zero_hidden_states.device)  # 假设为 float 类型
            
            # 例如，将其扩展为 (batch_size, 1, hidden_dim)
            i_embeds = i_tensor.unsqueeze(-1).repeat(1, 1, zero_hidden_states.size(-1))  # (batch_size, 1, hidden_dim)
            # Step 4: 计算 combined_embeds
            if i == 1:
                ar_ids = torch.argmax(ar_logits, dim=-1).long()
                audio_embeds = self.audio_input_embedding_layers2[i-1](ar_ids)#我使用第0通道以及第0的embed
            else:
                audio_embeds = self.audio_input_embedding_layers2[i-1](audio_input_ids[:,i-1,:]) #[0-9] -> [1-9]

            combined_embeds = (zero_hidden_states + audio_embeds) / 2.0  # 形状为 (batch_size, lg, hidden_dim)
            new_combined_embeds = torch.cat([i_embeds, combined_embeds], dim=1)  # 形状为 (batch_size, lg + 1, hidden_dim)

            i_attention_mask = torch.full((batch_size, 1), 1, dtype=zero_hidden_states.dtype, device=zero_hidden_states.device).long()  # 假设为 float 类型
            q_attention_mask = torch.cat([i_attention_mask, attention_mask], dim=1)
            q_position_ids = (q_attention_mask.cumsum(-1) - 1).masked_fill_((q_attention_mask == 0), 1)
            
            # Step 5: 将 i_embeds 拼接到 combined_embeds 前面，得到新的 combined_embeds
            #new_combined_embeds = torch.cat([i_embeds, combined_embeds], dim=1)  # 形状为 (batch_size, lg + 1, hidden_dim)
            CA_hidden_states = self.depformer_layers(new_combined_embeds, attention_mask=q_attention_mask,position_ids=q_position_ids)
            logits_audio = self.nar_lmhead(CA_hidden_states[:,1:,:])
        # if training:
        #     zero_hidden_states = self.shift_really_value(hidden_states,audio_labels) # [1-10] -> [1-9]
        #     i = random.randint(0, 5)
        #     i_tensor = torch.full((batch_size, 1), i+1, dtype=zero_hidden_states.dtype, device=zero_hidden_states.device)  # 假设为 float 类型
        #     #i_label = torch.full((batch_size, quantization, 1), IGNORE_INDEX, dtype=zero_hidden_states.dtype, device=zero_hidden_states.device).long()  # 假设为 float 类型
            
        #     # 例如，将其扩展为 (batch_size, 1, hidden_dim)
        #     i_embeds = i_tensor.unsqueeze(-1).repeat(1, 1, zero_hidden_states.size(-1))  # (batch_size, 1, hidden_dim)
        #     # Step 4: 计算 combined_embeds
        #     if i == 0:
        #         ar_quantization  = torch.argmax(ar_logits, dim=-1).long()
        #         ar_ids = self.shift_really_ids(ar_quantization,audio_labels) #[1-10] -> [1-9]
        #         audio_embeds = self.audio_input_embedding_layers2[i](ar_ids)
        #         new_combined_embeds = torch.cat([i_embeds, audio_embeds], dim=1)  # 形状为 (batch_size, lg + 1, hidden_dim)
        #     else:
        #         audio_embeds = self.audio_input_embedding_layers2[i](audio_input_ids[:,i,1:]) #[0-9] -> [1-9]
        #         combined_embeds = (zero_hidden_states + audio_embeds) / 2.0  # 形状为 (batch_size, lg, hidden_dim)
        #         new_combined_embeds = torch.cat([i_embeds, combined_embeds], dim=1)  # 形状为 (batch_size, lg + 1, hidden_dim)

        #     q_attention_mask = attention_mask#[:,1:] #[1-10] -> [1-9]
        #     q_position_ids = (q_attention_mask.cumsum(-1) - 1).masked_fill_((q_attention_mask == 0), 1)
            
        #     # Step 5: 将 i_embeds 拼接到 combined_embeds 前面，得到新的 combined_embeds
        #     #new_combined_embeds = torch.cat([i_embeds, combined_embeds], dim=1)  # 形状为 (batch_size, lg + 1, hidden_dim)
        #     CA_hidden_states = self.depformer_layers(new_combined_embeds, attention_mask=q_attention_mask,position_ids=q_position_ids)
        #     logits_audio = self.nar_lmhead(CA_hidden_states[:,1:,:])
      
        else:
            zero_hidden_states = hidden_states
            for i in range(0,quantization):
                if i==0:
                    logits_inter = ar_logits
                    logits_audio = ar_logits.unsqueeze(1)
                else:
                    i_tensor = torch.full((batch_size, 1), i, dtype=zero_hidden_states.dtype, device=zero_hidden_states.device)  # 假设为 float 类型
                    i_embeds = i_tensor.unsqueeze(-1).repeat(1, 1, zero_hidden_states.size(-1))  # (batch_size, 1, hidden_dim)
                    next_quantization_token  = torch.argmax(logits_inter, dim=-1).long()
                    i_attention_mask = torch.full((batch_size, 1), 1, dtype=zero_hidden_states.dtype, device=zero_hidden_states.device).long()  # 假设为 float 类型
                    q_attention_mask = torch.cat([i_attention_mask, attention_mask], dim=1)
                    q_position_ids = (q_attention_mask.cumsum(-1) - 1).masked_fill_((q_attention_mask == 0), 1)
                    audio_embeds = self.audio_input_embedding_layers2[i-1](next_quantization_token) #[0-9] -> [1-9]
                    combined_embeds = (zero_hidden_states + audio_embeds) / 2.0  # 形状为 (batch_size, lg, hidden_dim)
                    new_combined_embeds = torch.cat([i_embeds, combined_embeds], dim=1)  # 形状为 (batch_size, lg + 1, hidden_dim)
                    CA_hidden_states = self.depformer_layers(new_combined_embeds,attention_mask=q_attention_mask,position_ids=q_position_ids)
                    logits_inter = self.nar_lmhead(CA_hidden_states[:,1:,:])
                    logits_audio = torch.cat([logits_audio, logits_inter.unsqueeze(1)], dim=1) #[bs,q,lg,card]

        total_loss = 0.0
###################################################################################
        if training:
            logits_q = ar_logits[..., :, :]
            labels_q = audio_labels[..., 0, :]
            loss_audio = self.loss_fct_audio(logits_q, labels_q)
            total_loss += 1.0*loss_audio
            self.loss_fct_q = nn.CrossEntropyLoss(ignore_index=-100)
            logits_q = logits_audio 
            label_q = audio_labels[..., i, :] 
            if i == 1:
                loss_q = self.loss_fct_q(logits_q[..., :-1, :].contiguous().reshape(-1, self.card),label_q[..., 1:].contiguous().reshape(-1))
            else:
                loss_q = self.loss_fct_q(logits_q[..., :, :].contiguous().reshape(-1, self.card),label_q[..., :].contiguous().reshape(-1))
            total_loss += 1.0*loss_q
###################################################################################        
            total_loss = total_loss / 2.0
        logits = {
            "audio_logits": logits_audio,
            "text_logits": logits_text
        }

        return {
            "loss": total_loss if total_loss != 0.0 else None,
            "logits": logits,
            "hidden_states": outputs.hidden_states if outputs else None,
            "past_key_values": outputs.past_key_values if outputs else None,
            "attention_mask": attention_mask
        }
    import torch

    def generate_loop(
            self,
            max_length: int = 150,
            temperature: float = 1.0,
            audio_input_ids: Optional[torch.Tensor] = None,
            full_inputs: Optional[torch.Tensor] = None,
            stream: Optional[torch.Tensor] = None,
        ):
        """
        Generate both audio and text tokens based on audio and initial text inputs.

        Args:
            audio_input_ids (torch.Tensor): Audio input tokens. Shape: [batch_size, quantization, seq_len]
            text_input_ids (torch.Tensor): Initial text input tokens. Shape: [batch_size, initial_seq_len]
            max_length (int): Maximum length of the generated sequence.
            temperature (float): Sampling temperature.

        Returns:
            Dict[str, torch.Tensor]: Generated audio and text tokens.
        """
        self.model.eval()
        generated_text = full_inputs["input_ids"]
        attn_mask = full_inputs["attention_mask"]
        print("Initial attention mask size:", attn_mask.size())
        batch_size = generated_text.size(0)
        prefill_token = generated_text.size(-1)
        print("Initial prefill_token size:", prefill_token)
        # Initialize audio_input_ids
        audio_input_ids = torch.full((batch_size, 7, generated_text.size(-1)), 1026, device=generated_text.device, dtype=torch.long)
        
        # Initialize finished mask
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated_text.device)
        # 获取 eos_token_id
        eos_token_id = self.config_kwargs['eos_token_id']
        #pad_token_id = self.processor.pad_token_id
        with torch.no_grad():
            for step in range(max_length):
                #print(generated_text)
                #print(audio_input_ids)
                outputs = self.forward(
                    input_features=full_inputs["input_features"],
                    text_input_ids=generated_text,
                    text_labels=generated_text,
                    audio_input_ids=audio_input_ids,
                    audio_labels=audio_input_ids,
                    attention_mask=attn_mask,   # Pass dynamic attention mask
                    feature_attention_mask=full_inputs["feature_attention_mask"],
                    training=False
                )
                # Update attention mask
                new_attn_mask = torch.ones((attn_mask.size(0), 1), device=attn_mask.device)
                attn_mask = torch.cat([attn_mask, new_attn_mask], dim=1).long()
                # Get logits
                logits_text = outputs["logits"]["text_logits"][:, -1, :] / temperature
                logits_audio = outputs["logits"]["audio_logits"][:, :, -1, :] / temperature  # Shape: [batch_size, quantization, vocab_size]
                # Sample next text token
                probabilities_text = torch.softmax(logits_text, dim=-1)
                next_text_token = torch.multinomial(probabilities_text, num_samples=1)  # Shape: [batch_size, 1]
                probabilities_text  = torch.argmax(logits_text, dim=-1).long()
                next_text_token = probabilities_text.unsqueeze(1)  # Shape: [batch_size, 1]
                # Determine which batch elements have generated EOS
                eos_generated = (next_text_token.squeeze(-1) == eos_token_id)
                finished = finished | eos_generated  # Update finished mask
                # Create EOS tensor
                eos_tensor = torch.full((batch_size, 1), eos_token_id, dtype=generated_text.dtype, device=generated_text.device)
                # For finished examples, append EOS; for others, append the generated token
                next_text_token = torch.where(
                    finished.unsqueeze(-1),
                    eos_tensor,
                    next_text_token
                )
                #######################################
                # probabilities_text  = torch.argmax(logits_text, dim=-1).long()
                # next_generated_token = probabilities_text.unsqueeze(1)  # Shape: [batch_size, 1]
                # # Determine which batch elements have generated EOS
                # eos_generated = (next_generated_token.squeeze(-1) == eos_token_id)
                # finished = finished | eos_generated  # Update finished mask
                # # Create EOS tensor
                # eos_tensor = torch.full((batch_size, 1), eos_token_id, dtype=generated_text.dtype, device=generated_text.device)
                # # For finished examples, append EOS; for others, append the generated token
                # pad_tensor = torch.full(
                #     (batch_size, 1),
                #     pad_token_id,
                #     dtype=generated_text.dtype,
                #     device=generated_text.device
                # )
                
                # # Set next_text_token to PAD where finished is True
                # # Then set to EOS where EOS was generated in this step
                # next_text_token = torch.where(
                #     finished.unsqueeze(-1),
                #     pad_tensor,          # Set to PAD if already finished
                #     next_generated_token # Else, use the generated token
                # )
                # next_text_token = torch.where(
                #     eos_generated.unsqueeze(-1),
                #     eos_tensor,          # Override to EOS if just generated EOS
                #     next_text_token      # Else, keep as is (either PAD or generated token)
                # )
                #######################################
                # Append the token
                generated_text = torch.cat((generated_text, next_text_token), dim=-1)

                # Log if any new EOS tokens were generated in this step
                if torch.any(eos_generated):
                    logger.info(f"Step {step}: End-of-sequence token generated for {torch.sum(eos_generated).item()} examples.")
                #     #test2
                if step ==0:
                   next_audio_tokens = torch.full((batch_size, 7, 1),1024, device=generated_text.device, dtype=torch.long)
                else:
                    # Sample next audio tokens
                    next_audio_tokens = torch.zeros(batch_size, 7, 1, device=generated_text.device, dtype=torch.long)
                    for q in range(7):
                        logits_audio_q = logits_audio[:, q, :]  # Shape: [batch_size, vocab_size]
                        # 使用 argmax 采样
                        next_audio_token_q = torch.argmax(logits_audio_q, dim=-1).long()
                        next_audio_token_q = next_audio_token_q.unsqueeze(1)  # Shape: [batch_size, 1]
                        # next_audio_token_q = torch.softmax(logits_audio_q, dim=-1)
                        # next_audio_token_q = torch.multinomial(next_audio_token_q, num_samples=1)  # Shape: [batch_size, 1]
                        next_audio_tokens[:, q, :] = next_audio_token_q

                # Append audio tokens
                audio_input_ids = torch.cat((audio_input_ids, next_audio_tokens), dim=-1)   
                # Check if 2049 is generated in next_audio_tokens
                if torch.any(next_audio_tokens == 1025):
                    logger.info(f"Step {step}: End-of-sequence token (1025) generated in audio.")
                    break

                if stream:
                    mask = (next_audio_tokens == 1024) | (next_audio_tokens == 1025) | (next_audio_tokens == 1026)
                    next_audio_tokens[mask] = 0
                    original_discrete_code = []
                    combined = next_audio_tokens.permute(0, 2, 1)  # [batch_size, L, 7]
                    # 拆分出三个张量
                    tensor1_grouped = combined[:, :, 0:1]  # [batch_size, L, 1]
                    tensor2_grouped = combined[:, :, 1:3]  # [batch_size, L, 2]
                    tensor3_grouped = combined[:, :, 3:7]  # [batch_size, L, 4]

                    # 还原原始的张量形状
                    tensor1 = tensor1_grouped.squeeze(-1)  # [batch_size, L]
                    tensor2 = tensor2_grouped.contiguous().view(batch_size, -1)  # [batch_size, 2L]
                    tensor3 = tensor3_grouped.contiguous().view(batch_size, -1)  # [batch_size, 4L]
                    original_discrete_code = [
                        tensor1.to(next_text_token.device),
                        tensor2.to(next_text_token.device),
                        tensor3.to(next_text_token.device)]
                    #mask = (audio_input_ids != 2048) & (audio_input_ids != 2049)  # Create a mask where values are not 2048 or 2049
                    #audio_input_ids = audio_input_ids.masked_select(mask).view(batch_size, 8, -1)#.view(batch_size, 8, -1)  # Apply the mask
                    yield {
                        "audio_tokens": original_discrete_code,  # Shape: [batch_size, quantization, gen_seq_len]
                        "text_tokens": next_text_token,  # Shape: [batch_size, gen_seq_len]
                    }
                    
        if stream==False:
            audio_input_ids = audio_input_ids[:,:,prefill_token:]
            mask = (audio_input_ids == 1024) | (audio_input_ids == 1025) | (audio_input_ids == 1026)
            audio_input_ids[mask] = 0
            original_discrete_code = []
            combined = audio_input_ids.permute(0, 2, 1)  # [batch_size, L, 7]
            # 拆分出三个张量
            tensor1_grouped = combined[:, :, 0:1]  # [batch_size, L, 1]
            tensor2_grouped = combined[:, :, 1:3]  # [batch_size, L, 2]
            tensor3_grouped = combined[:, :, 3:7]  # [batch_size, L, 4]

            # 还原原始的张量形状
            tensor1 = tensor1_grouped.squeeze(-1)  # [batch_size, L]
            tensor2 = tensor2_grouped.contiguous().view(batch_size, -1)  # [batch_size, 2L]
            tensor3 = tensor3_grouped.contiguous().view(batch_size, -1)  # [batch_size, 4L]
            original_discrete_code = [
                tensor1.to(generated_text.device),
                tensor2.to(generated_text.device),
                tensor3.to(generated_text.device)]
            #mask = (audio_input_ids != 2048) & (audio_input_ids != 2049)  # Create a mask where values are not 2048 or 2049
            #audio_input_ids = audio_input_ids.masked_select(mask).view(batch_size, 8, -1)#.view(batch_size, 8, -1)  # Apply the mask
            yield {
                "audio_tokens": original_discrete_code,  # Shape: [batch_size, quantization, gen_seq_len]
                "text_tokens": generated_text  # Shape: [batch_size, gen_seq_len]
            }

