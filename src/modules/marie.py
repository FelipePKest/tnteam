"""Compatibility imports for MARIE. Implementations live in agents, critics,
and world_models/marie; existing imports and checkpoint class names still work.
"""

import copy
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.matwm import MATWMPolicy, MATWMWorldModel
from modules.vector_quantizer import EMAVectorQuantizer
from modules.world_models.marie.attention import _FixedKVCache, CachedCausalTransformerLayer, CachedCausalTransformer
from modules.world_models.marie.tokenizer import MARIEVQTokenizer
from modules.world_models.marie.aggregation import _GEGLU, _PerceiverAttention, _PerceiverFeedForward, PerceiverAggregator
from modules.world_models.marie.world_model import MARIEWorldModel
from modules.agents.marie_actor import OriginalMARIEActor
from modules.critics.marie import OriginalMARIECritic
from modules.agents.marie_policy import MARIEPolicy
