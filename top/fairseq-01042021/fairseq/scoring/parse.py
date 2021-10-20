from fairseq.scoring import BaseScorer, register_scorer


@register_scorer("parse")
class ParseScorer(BaseScorer):
    def __init__(self, args):
        super(ParseScorer, self).__init__(args)

        self.sacrebleu = sacrebleu

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.pred.append(pred)

    def score(self, order=4):
        return 100

    def result_string(self):
        import pdb; pdb.set_trace()
        return f"Parse: {self.score():.2f}"
