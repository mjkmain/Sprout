from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import torch

@dataclass
class SproutPTDataCollator:
    tokenizer: PreTrainedTokenizer
    
    """
    DataCollator For LLM Pretraining
    """
    
@dataclass
class SproutSFTDataCollator:
    tokenizer: PreTrainedTokenizer
    
    """
    DataCollator For LLM Supervised Fine-Tuning
    """