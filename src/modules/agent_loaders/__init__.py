REGISTRY = {}

from .rnn_eval_agent_loader import RNNEvalAgentLoader
from .rnn_train_agent_loader import RNNTrainAgentLoader
from .poam_eval_agent_loader import POAMEvalAgentLoader
from .poam_train_agent_loader import POAMTrainAgentLoader
from .type_conditional_loader import TypeConditionalAgentLoader
from .type_matched_train_loader import TypeMatchedTrainLoader
from .clam_train_agent_loader import CLAMTrainAgentLoader
from .matwm_train_agent_loader import MATWMTrainAgentLoader
from .marie_agent_loader import (
    MARIEEvalAgentLoader, MARIETrainAgentLoader,
    MARIEReferenceTrainAgentLoader,
)


REGISTRY["rnn_eval_agent_loader"] = RNNEvalAgentLoader
REGISTRY["rnn_train_agent_loader"] = RNNTrainAgentLoader
REGISTRY["poam_eval_agent_loader"] = POAMEvalAgentLoader
REGISTRY["poam_train_agent_loader"] = POAMTrainAgentLoader
REGISTRY["type_conditional_loader"] = TypeConditionalAgentLoader
REGISTRY["type_matched_train_loader"] = TypeMatchedTrainLoader
REGISTRY["clam_train_agent_loader"] = CLAMTrainAgentLoader
REGISTRY["matwm_train_agent_loader"] = MATWMTrainAgentLoader
REGISTRY["marie_train_agent_loader"] = MARIETrainAgentLoader
REGISTRY["marie_eval_agent_loader"] = MARIEEvalAgentLoader
REGISTRY["marie_reference_train_agent_loader"] = MARIEReferenceTrainAgentLoader
