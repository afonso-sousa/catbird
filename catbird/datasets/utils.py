import torch


class TeacherForcing:
    def __init__(self, batch_first=True, token_id=None):
        self.batch_first = batch_first
        self.token_id = token_id

    def __call__(self, src, targets):
        inputs = targets
        if self.batch_first:
            if self.token_id is None:
                # if no token_id is specified, then use the last token in src
                eos_tensor = src[:, 0].reshape(
                    -1, 1
                )  # transpose shape [2] into shape [2, 1]
            else:
                eos_tensor = torch.full((inputs.size(0), 1), self.token_id)
            input_shifted = inputs[:, :-1]
            inputs = torch.cat((eos_tensor, input_shifted), dim=1)
        else:
            if self.token_id is None:
                eos_tensor = src[0].unsqueeze(0)
            else:
                eos_tensor = torch.full((1, inputs.size(1)), self.token_id)
            input_shifted = inputs[:-1]
            inputs = torch.cat((eos_tensor, input_shifted), dim=0)
        return src, inputs, targets
