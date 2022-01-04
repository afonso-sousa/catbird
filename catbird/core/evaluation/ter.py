"""Ignite wrapper for HuggingFace TER implementation."""
from typing import Any, Callable, Sequence, Tuple, Union

import torch
from datasets import load_metric
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class TER(Metric):
    """Ignite wrapper for HuggingFace TER implementation."""

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        """Init class.

        Args:
            output_transform (Callable, optional): a callable required by the Ignite framework. Defaults to lambdax:x.
            device (Union[str, torch.device], optional): specifies which device updates are accumulated on. Defaults to torch.device("cpu").
        """
        super(TER, self).__init__(output_transform=output_transform, device=device)
        self._ter = load_metric("ter")

    def _sentence_ter(
        self, references: Sequence[Sequence[Any]], candidates: Sequence[Any]
    ) -> float:
        """Compute per-sentence TER score.

        Args:
            references (Sequence[Sequence[Any]]): Reference sentences.
            candidates (Sequence[Any]): Candidate sentence.

        Returns:
            float: TER score.
        """
        results = self._ter.compute(predictions=[candidates], references=[references])
        return results["score"]

    def reset(self) -> None:
        """Reset the metric to it's initial state.

        By default, this is called at the start of each epoch.
        """
        self._sum_of_ter = torch.tensor(0.0, dtype=torch.double, device=self._device)
        self._num_sentences = 0

    def update(
        self, output: Tuple[Sequence[Sequence[Any]], Sequence[Sequence[Sequence[Any]]]]
    ) -> None:
        """
        Update the metric's state using the passed batch output.

        By default, this is called once for each batch.
        Args:
            output: the output from the engine's process function.
        """
        y_pred, y = output

        for refs, hyp in zip(y, y_pred):
            self._sum_of_ter += self._sentence_ter(references=refs, candidates=hyp)
            self._num_sentences += 1

    def compute(self) -> Any:
        """Compute the metric based on it's accumulated state.

        By default, this is called at the end of each epoch.

        Raises:
            NotComputableError: raised when the metric cannot be computed.

        Returns:
            Any: the TER score
        """
        if self._num_sentences == 0:
            raise NotComputableError(
                "TER must have at least one example before it can be computed."
            )

        return self._sum_of_ter / self._num_sentences


if __name__ == "__main__":
    y_pred = "the the the the the the the"
    y = ["the cat is on the mat", "there is a cat on the mat"]

    y_pred1 = [y_pred.split()]
    y1 = [[_y.split() for _y in y]]

    ter = load_metric("ter")
    results = ter.compute(predictions=y_pred1, references=y1)

    my_ter = TER()
    my_ter.update((y_pred1, y1))

    assert round(my_ter.compute().item(), 4) == round(
        ter.compute(predictions=y_pred1, references=y1)["score"], 4
    )

    print(round(my_ter.compute().item(), 4))
