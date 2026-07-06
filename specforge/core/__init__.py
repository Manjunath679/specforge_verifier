from .dflash import OnlineDFlashModel
from .domino import OnlineDominoModel
from .eagle3 import OnlineEagle3Model, QwenVLOnlineEagle3Model
# TEST TIME EAGLE EXP: opt-in hidden adapter export; original EAGLE3 flow is unchanged.
from .eagle3_hidden_adapter_pipeline import (
    GatedResidualHiddenAdapter,
    HiddenAdapterDraftWrapper,
    OnlineEagle3HiddenAdapterModel,
)
from .eagle3_verifier import Eagle3CandidateVerifier, OnlineEagle3VerifierModel
from .peagle import OnlinePEagleModel

__all__ = [
    "Eagle3CandidateVerifier",
    "GatedResidualHiddenAdapter",
    "HiddenAdapterDraftWrapper",
    "OnlineDFlashModel",
    "OnlineDominoModel",
    "OnlineEagle3Model",
    "OnlineEagle3HiddenAdapterModel",
    "OnlineEagle3VerifierModel",
    "OnlinePEagleModel",
    "QwenVLOnlineEagle3Model",
]
