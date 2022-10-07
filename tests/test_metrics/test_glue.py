from datasets import load_metric
from ignite.metrics.nlp import Bleu
from catbird.core.evaluation import Meteor, TER
import pytest
from nltk import word_tokenize
from ignite.engine import Engine
from ignite.metrics import Accuracy


def preproc(text, lower_split=True):
    if lower_split:
        return text.lower().split()
    else:
        return text


cand_1 = preproc("the the the the the the the")
ref_1a = preproc("The cat is on the mat")
ref_1b = preproc("There is a cat on the mat")

cand_2a = preproc(
    "It is a guide to action which ensures that the military always obeys the commands of the party"
)
cand_2b = preproc(
    "It is to insure the troops forever hearing the activity guidebook that "
    "party direct"
)
ref_2a = preproc(
    "It is a guide to action that ensures that the military will forever heed "
    "Party commands"
)
ref_2b = preproc(
    "It is the guiding principle which guarantees the military forces always being under the command of "
    "the Party"
)
ref_2c = preproc(
    "It is the practical guide for the army always to heed the directions of the party"
)

cand_3 = preproc("of the")

references_1 = [ref_1a, ref_1b]
references_2 = [ref_2a, ref_2b, ref_2c]

# @pytest.mark.parametrize("metric", [Bleu, Meteor, TER])
def test_meteor_against_datasets_metric():
    y_pred = "the the the the the the the"
    y = ["the cat is on the mat", "there is a cat on the mat"]

    y_pred1 = [y_pred.split()]
    y1 = [[_y.split() for _y in y]]

    # m = Bleu(ngram=4, smooth="smooth1")
    # m.update((y_pred1, y_pred1))
    # print(m.compute())

    meteor = load_metric("meteor")
    predictions = [
        "It is a guide to action which ensures that the military always obeys the commands of the party"
    ]
    references = [
        "It is a guide to action which ensures that the military always obeys the commands of the party"
    ]
    results = meteor.compute(predictions=predictions, references=references)
    print(round(results["meteor"], 4))

    # assert False

    meteor = load_metric("meteor")

    my_meteor = Meteor()
    my_meteor.update((predictions, references))
    print(my_meteor.compute())

    assert round(my_meteor.compute().item(), 4) == round(
        meteor.compute(predictions=predictions, references=references)["meteor"], 4
    )

    m = Bleu(ngram=4, smooth="smooth1")
    m.update(([predictions[0].split()], [[references[0].split()]]))
    assert m.compute().item() == 1.0

    print(m.compute().item())

    my_meteor.update((y_pred1, [y_pred1]))
    assert my_meteor.compute().item() == 1.0


def test_with_engine():
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

    # Batch size 3
    hypotheses = [cand_1, cand_2a, cand_2b]
    refs = [references_1, references_2, references_2]

    print(hypotheses)
    print(refs)

    bleu = Bleu(ngram=4)
    bleu.update((hypotheses, refs))
    print(bleu.compute())

    batch1 = [
        ["the the the the cat", "the the the the", "mat patmat patmat pat cat"],
        ["the the the the dog", "the cat the the the", "mat patmat patmat pat cat dog"],
        # ["the the the cat", "the the the the", "mat patmat patmat pat"],
    ]

    data = [batch1, batch1]

    def process_function(engine, batch):
        y_pred = [word_tokenize(_y_pred) for _y_pred in batch[0]]
        y = [[word_tokenize(_y)] for _y in batch[1]]
        print(y_pred, y)
        return y_pred, y

    t = process_function(None, batch1)
    bleu = Bleu(ngram=4)
    bleu.update((t[0], t[1]))
    print(bleu.compute())

    print(t[0][0])
    print(t[1][0])
    print(sentence_bleu(t[1][0], t[0][0]))
    print(sentence_bleu(t[1][1], t[0][1]))
    print(sentence_bleu(t[1][2], t[0][2]))

    assert False

    engine = Engine(process_function)
    metric = Bleu(ngram=4)
    metric.attach(engine, "bleu")

    state = engine.run(data)
    print(f"BLEU: {state.metrics['bleu']}")

    assert False
