from typing import Any, Callable, Sequence, Tuple, Union

import nltk
import torch
from datasets import load_metric
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class GLUE(Metric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(GLUE, self).__init__(output_transform=output_transform, device=device)
        self._ter = load_metric("ter")

    def _sentence_ter(
        self, references: Sequence[Sequence[Any]], candidates: Sequence[Any]
    ) -> float:
        results = self._ter.compute(predictions=[candidates], references=[references])
        return results["score"]

    def reset(self) -> None:
        self._sum_of_ter = torch.tensor(0.0, dtype=torch.double, device=self._device)
        self._num_sentences = 0

    def update(
        self, output: Tuple[Sequence[Sequence[Any]], Sequence[Sequence[Sequence[Any]]]]
    ) -> None:
        y_pred, y = output

        for refs, hyp in zip(y, y_pred):
            self._sum_of_ter += self._sentence_ter(references=refs, candidates=hyp)
            self._num_sentences += 1

    def compute(self) -> None:
        if self._num_sentences == 0:
            raise NotComputableError(
                "TER must have at least one example before it can be computed."
            )

        return self._sum_of_ter / self._num_sentences


if __name__ == "__main__":
    from datasets import list_metrics
    from ignite.metrics.nlp import Bleu

    metrics_list = list_metrics()
    print(", ".join(metric for metric in metrics_list))

    y_pred = "the the the the the the the"
    y = ["the cat is on the mat", "there is a cat on the mat"]

    y_pred1 = [y_pred.split()]
    y1 = [[_y.split() for _y in y]]

    m = Bleu(ngram=4, smooth="smooth1")
    m.update((y_pred1, y1))
    print(m.compute())

    from datasets import load_metric

    ter = load_metric("ter")
    print(ter.inputs_description)

    results = ter.compute(predictions=y_pred1, references=y1)

    print(round(results["score"], 4))

    my_ter = TER()
    my_ter.update((y_pred1, y1))
    print(my_ter.compute())

    assert round(my_ter.compute(), 4) == round(
        ter.compute(predictions=y_pred1, references=y1)["score"], 4
    )

    print(round(my_ter.compute().item(), 4))
