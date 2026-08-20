from evaluation.evaluate_retrieval_compare import calculate_metrics


class Document:
    def __init__(self, text):
        self.page_content = text


def test_calculate_metrics_includes_ndcg():
    values = calculate_metrics(
        [[Document("target"), Document("noise")]],
        [{"keywords": ["target"]}],
    )
    assert values["Hit@1"] == 1.0
    assert values["MRR"] == 1.0
    assert values["NDCG@3"] == 1.0


def test_ndcg_penalizes_late_relevant_result():
    values = calculate_metrics(
        [[Document("noise"), Document("target")]],
        [{"keywords": ["target"]}],
    )
    assert 0.0 < values["NDCG@3"] < 1.0
