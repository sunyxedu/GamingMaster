"""model.optim.rewarm_lr_lambda: a run continued past a finished WSD decay ramps its rate from
the decayed floor back to the schedule's over --rewarmup_steps instead of jumping."""
from model.optim import rewarm_lr_lambda, wsd_lr_lambda


def test_rewarm_ramps_from_the_checkpoint_ratio_to_the_schedule():
    old = wsd_lr_lambda(2000, 90000, 0.1, 0.1, "linear")        # the finished run
    assert abs(old(90000) - 0.1) < 1e-9 and old(50000) == 1.0
    new = wsd_lr_lambda(2000, 140000, 0.1, 0.1, "linear")       # --steps raised: constant at 90k
    assert new(90001) == 1.0
    fn = rewarm_lr_lambda(new, 90000, old(90000), 2000)
    assert fn(90000) == new(90000)                              # untouched up to the resume step
    assert abs(fn(90001) - (0.1 + 0.9 / 2000)) < 1e-9           # starts at the floor
    assert abs(fn(91000) - 0.55) < 1e-9                         # half way
    assert fn(92000) == 1.0 and fn(100000) == 1.0               # the schedule's own value after
    assert abs(fn(135000) - new(135000)) < 1e-12 and new(135000) < 1.0   # the new decay as is
    assert rewarm_lr_lambda(new, 90000, 0.1, 0) is new           # off
