"""
Some hypothesis sample tests.

See http://hypothesis.works/
"""
# 3rd party
from hypothesis import given
from hypothesis import strategies as st

# My stuff
from samples.sample_module import reverse_manually


@given(
    a_string=st.text()
)
def test_reverse(a_string):
    """
    `reverse(reverse(x)) == x` is a perfect hypothesis test sample.

    Since it must be true for every given string `x`.

    See also http://hypothesis.works/articles/encode-decode-invariant/.
    """
    assert reverse_manually(reverse_manually(a_string)) == a_string
