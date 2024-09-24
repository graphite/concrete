#  Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
#  See https://github.com/zama-ai/concrete/blob/main/LICENSE.txt for license information.

"""Compiler submodule."""
import atexit
from typing import Union

# pylint: disable=no-name-in-module,import-error
from mlir._mlir_libs._concretelang._compiler import (
    terminate_df_parallelization as _terminate_df_parallelization,
    init_df_parallelization as _init_df_parallelization,
    check_gpu_runtime_enabled as _check_gpu_runtime_enabled,
    check_cuda_device_available as _check_cuda_device_available,
)
from mlir._mlir_libs._concretelang._compiler import round_trip as _round_trip
from mlir._mlir_libs._concretelang._compiler import (
    set_llvm_debug_flag,
    set_compiler_logging,
    LweSecretKeyParam,
    BootstrapKeyParam,
    KeyswitchKeyParam,
    PackingKeyswitchKeyParam,
    ProgramInfo,
    ProgramCompilationFeedback,
    CircuitCompilationFeedback,
    CompilationOptions,
    CompilationContext,
    KeysetCache,
    ServerKeyset,
    Library,
    Keyset,
    Compiler,
    TransportValue,
    Value,
    PublicArguments,
    PublicResults,
    ServerProgram,
    ServerCircuit,
    ClientProgram,
    ClientCircuit
)

# pylint: enable=no-name-in-module,import-error

from .compilation_options import Encoding
from .utils import lookup_runtime_lib

from .tfhers_int import (
    TfhersExporter,
    TfhersFheIntDescription,
)

type Parameter = Union[LweSecretKeyParam, BootstrapKeyParam, KeyswitchKeyParam, PackingKeyswitchKeyParam]

def init_dfr():
    """Initialize dataflow parallelization.

    It is not always required to initialize the dataflow runtime as it can be implicitely done
    during compilation. However, it is required in case no compilation has previously been done
    and the runtime is needed"""
    _init_df_parallelization()


def check_gpu_enabled() -> bool:
    """Check whether the compiler and runtime support GPU offloading.

    GPU offloading is not always available, in particular in non-GPU wheels."""
    return _check_gpu_runtime_enabled()


def check_gpu_available() -> bool:
    """Check whether a CUDA device is available and online."""
    return _check_cuda_device_available()


# Cleanly terminate the dataflow runtime if it has been initialized
# (does nothing otherwise)
atexit.register(_terminate_df_parallelization)


def round_trip(mlir_str: str) -> str:
    """Parse the MLIR input, then return it back.

    Useful to check the validity of an MLIR representation

    Args:
        mlir_str (str): textual representation of an MLIR code

    Raises:
        TypeError: if mlir_str is not of type str

    Returns:
        str: textual representation of the MLIR code after parsing
    """
    if not isinstance(mlir_str, str):
        raise TypeError(f"mlir_str must be of type str, not {type(mlir_str)}")
    return _round_trip(mlir_str)
