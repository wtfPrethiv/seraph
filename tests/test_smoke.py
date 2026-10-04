from seraph.smoke import run


def test_smoke_passes(capsys):
    assert run() == 0
    assert "7/7 checks passed" in capsys.readouterr().out
