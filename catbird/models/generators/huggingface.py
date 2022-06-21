import ignite.distributed as idist
import torch
from torch import nn
from transformers import T5ForConditionalGeneration

from ..registry import GENERATORS
from ..modules import freeze_params


@GENERATORS.register_module
class HuggingFaceWrapper(nn.Module):
    def __init__(self, name, vocab_size, **kwargs):
        super(HuggingFaceWrapper, self).__init__()
        if name == "t5-small":
            model = T5ForConditionalGeneration.from_pretrained(name)
            model.resize_token_embeddings(vocab_size)
        else:
            raise NotImplementedError(
                "The name of the model you specified is not supported."
            )
        if kwargs.get("freeze_encoder", None):
            freeze_params(model.get_encoder())
        self.model = idist.auto_model(model)

    def forward(self, input_ids, attention_mask, tgt, return_loss=True, **kwargs):
        outputs = self.model.forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=tgt
        )
        loss, logits = outputs[:2]
        if return_loss:
            return loss
        else:
            return logits

    def generate(self, input_ids, **kwargs):
        return self.model.generate(input_ids=input_ids, **kwargs)
        # return torch.argmax(out, dim=-1)
