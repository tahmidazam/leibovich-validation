from leibovich_validation import progress


def test_bar_reports_done_total_and_postfix(capsys):
    with progress.bar(total=2, desc="hashing", disable=False) as bar:
        bar.set_postfix(file="a.csv", refresh=False)
        bar.update()
    err = capsys.readouterr().err
    assert "hashing" in err
    assert "1/2" in err
    assert "file=a.csv" in err
