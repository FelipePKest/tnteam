from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .poam_learner import POAMLearner
from .policy_repr_poam_learner import (
    PolicyRepresentationPOAMLearner as PolicyReprPOAMLearner,
)
from .clam_learner import CLAMLearner
from .clam_type_supervised_learner import TypeSupervisedCLAMLearner
from .classifier_learner import ClassifierLearner
from .policy_repr_poam_leaner import PolicyRepresentationPOAMLearner
from .matwm_learner import MATWMLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["maddpg_learner"] = MADDPGLearner
REGISTRY["ppo_learner"] = PPOLearner
REGISTRY["pac_learner"] = PACActorCriticLearner
REGISTRY["pac_dcg_learner"] = PACDCGLearner
REGISTRY["poam_learner"] = POAMLearner
REGISTRY["policy_repr_poam_learner"] = PolicyReprPOAMLearner
REGISTRY["clam_learner"] = CLAMLearner
REGISTRY["clam_type_supervised_learner"] = TypeSupervisedCLAMLearner
REGISTRY["classifier_learner"] = ClassifierLearner
REGISTRY["policy_repr_poam_learner"] = PolicyRepresentationPOAMLearner
REGISTRY["matwm_learner"] = MATWMLearner
