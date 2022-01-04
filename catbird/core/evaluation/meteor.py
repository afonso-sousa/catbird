"""Ignite wrapper for HuggingFace Meteor implementation."""
from typing import Any, Callable, Sequence, Tuple, Union

import nltk
import torch
from datasets import load_metric
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric

nltk.download("omw-1.4")


class Meteor(Metric):
    """Ignite wrapper for HuggingFace Meteor implementation."""

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
        super(Meteor, self).__init__(output_transform=output_transform, device=device)
        self._meteor = load_metric("meteor")

    def _sentence_meteor(
        self, references: Sequence[Sequence[Any]], candidates: Sequence[Any]
    ) -> float:
        """Compute per-sentence Meteor score.

        Args:
            references (Sequence[Sequence[Any]]): Reference sentences.
            candidates (Sequence[Any]): Candidate sentence.

        Returns:
            float: Meteor score.
        """
        results = self._meteor.compute(
            predictions=[candidates], references=[references]
        )
        return results["meteor"]

    def reset(self) -> None:
        """Reset the metric to it's initial state.

        By default, this is called at the start of each epoch.
        """
        self._sum_of_meteor = torch.tensor(0.0, dtype=torch.double, device=self._device)
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
            self._sum_of_meteor += self._sentence_meteor(
                references=refs, candidates=hyp
            )
            self._num_sentences += 1

    def compute(self) -> Any:
        """Compute the metric based on it's accumulated state.

        By default, this is called at the end of each epoch.

        Raises:
            NotComputableError: raised when the metric cannot be computed.

        Returns:
            Any: the Meteor score
        """
        if self._num_sentences == 0:
            raise NotComputableError(
                "Meteor must have at least one example before it can be computed."
            )

        return self._sum_of_meteor / self._num_sentences


if __name__ == "__main__":
    y_pred = "the the the the the the the"
    y = ["the cat is on the mat", "there is a cat on the mat"]

    y_pred1 = [y_pred.split()]
    y1 = [[_y.split() for _y in y]]

    meteor = load_metric("meteor")
    meteor_results = meteor.compute(predictions=y_pred1, references=y1)

    my_meteor = Meteor()
    my_meteor.update((y_pred1, y1))

    assert round(my_meteor.compute().item(), 4) == round(meteor_results["meteor"], 4)

    print(round(my_meteor.compute().item(), 4))
