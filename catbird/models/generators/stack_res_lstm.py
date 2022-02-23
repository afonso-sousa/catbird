from ..registry import GENERATORS
from .base import Seq2Seq


@GENERATORS.register_module
class StackedResidualLSTM(Seq2Seq):
    """This is an implementation of paper `Neural Paraphrase Generation
    with Stacked Residual LSTM Networks <https://aclanthology.org/C16-1275/>`.
    """

    def __init__(self, encoder, decoder):
        super(StackedResidualLSTM, self).__init__(encoder, decoder)
