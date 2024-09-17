from typing_extensions import NamedTuple
from concrete.compiler import ServerKeyset

class EvaluationKeys(NamedTuple):
    """
    EvaluationKeys required for execution.
    """

    server_keyset: ServerKeyset
