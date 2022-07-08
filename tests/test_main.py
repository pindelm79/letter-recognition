import pytest
from click.testing import CliRunner

from letter_recognition import __main__


@pytest.fixture
def runner():
    return CliRunner()


def test_main_succeeds(runner):
    result = runner.invoke(__main__.main)
    assert result.exit_code == 0
