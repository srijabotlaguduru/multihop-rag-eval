from collections import Counter

from ragbench.data import Question, sample


def _qs():
    return [Question(f"{h}-{i}", "q", "a", [], h, [0, 1]) for h, n in ((2, 50), (3, 30), (4, 20)) for i in range(n)]


def test_sample_is_stratified_by_hops():
    s = sample(_qs(), 30)
    assert Counter(q.hops for q in s) == {2: 10, 3: 10, 4: 10}


def test_sample_is_deterministic():
    assert [q.id for q in sample(_qs(), 30, seed=1)] == [q.id for q in sample(_qs(), 30, seed=1)]


def test_test_split_is_disjoint_from_dev():
    from ragbench.data import DEV_N, TEST_N, split

    qs = [Question(f"{h}-{i}", "q", "a", [], h, [0]) for h in (2, 3, 4) for i in range(200)]
    dev, test = split(qs, "dev"), split(qs, "test")
    assert len(dev) == DEV_N and len(test) == TEST_N
    assert not {q.id for q in dev} & {q.id for q in test}
    assert [q.id for q in dev] == [q.id for q in sample(qs, DEV_N, 13)]  # dev = the original sample
