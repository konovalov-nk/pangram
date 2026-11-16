import pytest
import torch

from ai_detector.detector import chunk_by_tokens, consensus_for, editlens_decode
from ai_detector.greyscope import decode_logits


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}


def test_chunk_by_tokens_covers_long_text_with_overlap():
    chunks = chunk_by_tokens("abcdefghij", CharacterTokenizer(), max_tokens=4, overlap=2)

    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]


def test_chunk_by_tokens_handles_empty_text():
    assert chunk_by_tokens("", CharacterTokenizer(), max_tokens=4) == []


def test_consensus_uses_majority_without_averaging_scores():
    result = consensus_for(
        [
            {"label": "human", "ai_probability": 0.001},
            {"label": "ai", "ai_probability": 0.51},
            {"label": "ai", "ai_probability": 0.52},
        ]
    )

    assert result == {
        "label": "ai",
        "ai_votes": 2,
        "human_votes": 1,
        "total": 3,
        "agreement": "majority",
    }


def test_consensus_reports_even_split_as_uncertain():
    result = consensus_for(
        [
            {"label": "human"},
            {"label": "ai"},
            {"label": "human"},
            {"label": "ai"},
        ]
    )

    assert result == {
        "label": "uncertain",
        "ai_votes": 2,
        "human_votes": 2,
        "total": 4,
        "agreement": "majority",
    }


def test_editlens_decode_maps_buckets_to_involvement_score():
    score, details = editlens_decode(torch.tensor([10.0, 0.0, 0.0, 0.0]))

    assert score == pytest.approx(0.0, abs=1e-3)
    assert details["score_kind"] == "ai_involvement"
    assert details["graded_label"] == "Human"
    assert details["bucket_probs"]["Human"] > 0.99


def test_editlens_decode_uniform_logits_are_half_involvement():
    score, details = editlens_decode(torch.zeros(4))

    assert score == pytest.approx(0.5)
    assert details["bucket_probs"] == pytest.approx(
        {
            "Human": 0.25,
            "Lightly AI-Assisted": 0.25,
            "Moderately AI-Assisted": 0.25,
            "AI-Generated": 0.25,
        }
    )


def test_greyscope_decodes_corn_logits():
    calibration = {
        "score_min": 0.0,
        "score_max": 1.0,
        "h_thresh": 0.1,
        "ai_thresh": 0.9,
        "bucket_descriptions": ["none", "light", "moderate", "heavy"],
    }

    result = decode_logits(torch.zeros(3), calibration)

    assert result["score"] == pytest.approx((0.5 + 0.25 + 0.125) / 3)
    assert result["graded_label"] == "AI-edited"
    assert result["bucket_probs"] == pytest.approx(
        {"none": 0.5, "light": 0.25, "moderate": 0.125, "heavy": 0.125}
    )
