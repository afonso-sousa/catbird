from torch import nn


class AddLossModule(nn.Module):
    """adds a loss to module for easy parallelization"""

    def __init__(self, module, criterion, ignore_index=PAD):
        super(AddLossModule, self).__init__()
        self.module = module
        self.criterion = criterion
        self.ignore_index = ignore_index

    def forward(self, module_inputs, target):
        output = self.module(*module_inputs)
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        output = nn.functional.log_softmax(output, -1)
        # make sure criterion is not from_logits
        loss = self.criterion(output, target).view(1, 1)
        nll = nn.functional.nll_loss(output, target,
                                     ignore_index=self.ignore_index,
                                     reduction='sum')

        _, argmax = output.max(-1)
        invalid_targets = target.eq(self.ignore_index)
        accuracy = argmax.eq(target).masked_fill_(invalid_targets, 0)\
            .long().sum()

        return loss, nll, accuracy.view(1, 1)


