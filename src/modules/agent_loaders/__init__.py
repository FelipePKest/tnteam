REGISTRY = {}

from .rnn_eval_agent_loader import RNNEvalAgentLoader
from .rnn_train_agent_loader import RNNTrainAgentLoader
from .poam_eval_agent_loader import POAMEvalAgentLoader
from .poam_train_agent_loader import POAMTrainAgentLoader
from .type_conditional_loader import TypeConditionalAgentLoader
from .type_matched_train_loader import TypeMatchedTrainLoader
from .clam_train_agent_loader import CLAMTrainAgentLoader
from .clam_eval_agent_loader import CLAMEvalAgentLoader
from .model_based_teammate_eval_agent_loader import ModelBasedTeammateEvalAgentLoader


REGISTRY["rnn_eval_agent_loader"] = RNNEvalAgentLoader
REGISTRY["rnn_train_agent_loader"] = RNNTrainAgentLoader
REGISTRY["poam_eval_agent_loader"] = POAMEvalAgentLoader
REGISTRY["poam_train_agent_loader"] = POAMTrainAgentLoader
REGISTRY["type_conditional_loader"] = TypeConditionalAgentLoader
REGISTRY["type_matched_train_loader"] = TypeMatchedTrainLoader
REGISTRY["clam_train_agent_loader"] = CLAMTrainAgentLoader
REGISTRY["clam_eval_agent_loader"] = CLAMEvalAgentLoader
REGISTRY["model_based_teammate_eval_agent_loader"] = ModelBasedTeammateEvalAgentLoader
