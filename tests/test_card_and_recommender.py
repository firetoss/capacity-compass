from capacity_compass import config_loader
from capacity_compass.pipeline.card_sizer import size_cards
from capacity_compass.pipeline.recommender import rank_candidates


def test_size_cards_prefers_compute_when_available() -> None:
    gpu = next(g for g in config_loader.load_hardware() if g.id == "nvidia_l20_48g")
    evaluation = size_cards(
        gpu=gpu,
        eval_precision="fp16",
        total_mem_bytes=40 * (2**30),
        required_compute_Tx=50,
    )
    assert evaluation.cards_needed >= evaluation.cards_mem


def test_recommender_orders_by_cards_price_support() -> None:
    hardware = config_loader.load_hardware()
    evaluations = [
        size_cards(hardware[0], "fp16", 10 * (2**30), 5),
        size_cards(hardware[1], "fp16", 10 * (2**30), 5),
    ]
    ranked = rank_candidates(evaluations)
    assert ranked
    assert ranked[0].cards_needed <= ranked[1].cards_needed
