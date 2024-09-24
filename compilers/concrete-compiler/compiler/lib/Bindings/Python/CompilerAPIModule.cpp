// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Bindings/Python/CompilerAPIModule.h"
#include "concrete-optimizer.hpp"
#include "concrete-protocol.capnp.h"
#include "concretelang/ClientLib/ClientLib.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Keys.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Values.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/GPUDFG.hpp"
#include "concretelang/ServerLib/ServerLib.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Error.h"
#include "concretelang/Support/V0Parameters.h"
#include "concretelang/Support/logging.h"
#include <memory>
#include <mlir-c/Bindings/Python/Interop.h>
#include <mlir/CAPI/IR.h>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/_types/_int8_t.h>

using concretelang::clientlib::ClientCircuit;
using concretelang::clientlib::ClientProgram;
using concretelang::keysets::Keyset;
using concretelang::keysets::KeysetCache;
using concretelang::keysets::ServerKeyset;
using concretelang::serverlib::ServerCircuit;
using concretelang::serverlib::ServerProgram;
using concretelang::values::TransportValue;
using concretelang::values::Value;
using mlir::concretelang::CompilationOptions;

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b

#define GET_OR_THROW_EXPECTED_(VARNAME, RESULT, MAYBE)                         \
  auto MAYBE = RESULT;                                                         \
  if (auto err = MAYBE.takeError()) {                                          \
    throw std::runtime_error(llvm::toString(std::move(err)));                  \
  }                                                                            \
  VARNAME = std::move(*MAYBE);

#define GET_OR_THROW_EXPECTED(VARNAME, RESULT)                                 \
  GET_OR_THROW_EXPECTED_(VARNAME, RESULT, CONCAT(maybe, __COUNTER__))

#define GET_OR_THROW_RESULT_(VARNAME, RESULT, MAYBE)                           \
  auto MAYBE = RESULT;                                                         \
  if (MAYBE.has_failure()) {                                                   \
    throw std::runtime_error(MAYBE.as_failure().error().mesg);                 \
  }                                                                            \
  VARNAME = MAYBE.value();

#define GET_OR_THROW_RESULT(VARNAME, RESULT)                                   \
  GET_OR_THROW_RESULT_(VARNAME, RESULT, CONCAT(maybe, __COUNTER__))

#define EXPECTED_TRY_(lhs, rhs, maybe)                                         \
  auto maybe = rhs;                                                            \
  if (auto err = maybe.takeError()) {                                          \
    return std::move(err);                                                     \
  }                                                                            \
  lhs = *maybe;

#define EXPECTED_TRY(lhs, rhs)                                                 \
  EXPECTED_TRY_(lhs, rhs, CONCAT(maybe, __COUNTER__))

template <typename T> llvm::Expected<T> outcomeToExpected(Result<T> outcome) {
  if (outcome.has_failure()) {
    return mlir::concretelang::StreamStringError(
        outcome.as_failure().error().mesg);
  } else {
    return outcome.value();
  }
}

namespace {
class SignalGuard {
public:
  SignalGuard() { previousHandler = signal(SIGINT, SignalGuard::handler); }
  ~SignalGuard() { signal(SIGINT, this->previousHandler); }

private:
  void (*previousHandler)(int);

  static void handler(int _signum) {
    llvm::outs() << " Aborting... \n";
    kill(getpid(), SIGKILL);
  }
};

void terminateDataflowParallelization() { _dfr_terminate(); }

void initDataflowParallelization() {
  mlir::concretelang::dfr::_dfr_set_required(true);
}

bool checkGPURuntimeEnabled() {
  return mlir::concretelang::gpu_dfg::check_cuda_runtime_enabled();
}

bool checkCudaDeviceAvailable() {
  return mlir::concretelang::gpu_dfg::check_cuda_device_available();
}

std::string roundTrip(const char *module) {
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::CompilerEngine ce{ccx};

  std::string backingString;
  llvm::raw_string_ostream os(backingString);

  llvm::Expected<mlir::concretelang::CompilerEngine::CompilationResult>
      retOrErr = ce.compile(
          module, mlir::concretelang::CompilerEngine::Target::ROUND_TRIP);
  if (!retOrErr) {
    os << "MLIR parsing failed: " << llvm::toString(retOrErr.takeError());
    throw std::runtime_error(os.str());
  }

  retOrErr->mlirModuleRef->get().print(os);
  return os.str();
}

// Every number sent by python through the API has a type `int64` that must be
// turned into the proper type expected by the ArgTransformers. This allows to
// get an extra transformer executed right before the ArgTransformer gets
// called.
std::function<Value(Value)>
getPythonTypeTransformer(const Message<concreteprotocol::GateInfo> &info) {
  if (info.asReader().getTypeInfo().hasIndex()) {
    return [=](Value input) {
      Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
      return Value{(Tensor<uint64_t>)tensorInput};
    };
  } else if (info.asReader().getTypeInfo().hasPlaintext()) {
    if (info.asReader().getTypeInfo().getPlaintext().getIntegerPrecision() <=
        8) {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint8_t>)tensorInput};
      };
    }
    if (info.asReader().getTypeInfo().getPlaintext().getIntegerPrecision() <=
        16) {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint16_t>)tensorInput};
      };
    }
    if (info.asReader().getTypeInfo().getPlaintext().getIntegerPrecision() <=
        32) {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint32_t>)tensorInput};
      };
    }
    if (info.asReader().getTypeInfo().getPlaintext().getIntegerPrecision() <=
        64) {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint64_t>)tensorInput};
      };
    }
    assert(false);
  } else if (info.asReader().getTypeInfo().hasLweCiphertext()) {
    if (info.asReader()
            .getTypeInfo()
            .getLweCiphertext()
            .getEncoding()
            .hasInteger() &&
        info.asReader()
            .getTypeInfo()
            .getLweCiphertext()
            .getEncoding()
            .getInteger()
            .getIsSigned()) {
      return [=](Value input) { return input; };
    } else {
      return [=](Value input) {
        Tensor<int64_t> tensorInput = input.getTensor<int64_t>().value();
        return Value{(Tensor<uint64_t>)tensorInput};
      };
    }
  } else {
    assert(false);
  }
};

template <typename T> Tensor<T> arrayToTensor(pybind11::array &input) {
  auto data_ptr = (const T *)input.data();
  std::vector<T> data = std::vector(data_ptr, data_ptr + input.size());
  auto dims = std::vector<size_t>(input.ndim(), 0);
  for (ssize_t i = 0; i < input.ndim(); i++) {
    dims[i] = input.shape(i);
  }
  return std::move(Tensor<T>(std::move(data), std::move(dims)));
}

template <typename T> pybind11::array tensorToArray(Tensor<T> input) {
  return pybind11::array(pybind11::array::ShapeContainer(input.dimensions),
                         input.values.data());
}
} // namespace

/// Populate the compiler API python module.
void mlir::concretelang::python::populateCompilerAPISubmodule(
    pybind11::module &m) {

  using ::concretelang::csprng::EncryptionCSPRNG;
  using ::concretelang::values::Value;
  using pybind11::arg;
  using pybind11::array;
  using pybind11::init;
  using Library = CompilerEngine::Library;

  m.doc() = "Concretelang compiler python API";

  m.def("round_trip",
        [](std::string mlir_input) { return roundTrip(mlir_input.c_str()); });

  m.def("set_llvm_debug_flag", [](bool enable) { llvm::DebugFlag = enable; });

  m.def("set_compiler_logging",
        [](bool enable) { mlir::concretelang::setupLogging(enable); });

  m.def("terminate_df_parallelization", &terminateDataflowParallelization);

  m.def("init_df_parallelization", &initDataflowParallelization);
  m.def("check_gpu_runtime_enabled", &checkGPURuntimeEnabled);
  m.def("check_cuda_device_available", &checkCudaDeviceAvailable);

  pybind11::class_<TfhersFheIntDescription>(m, "TfhersFheIntDescription")
      .def(pybind11::init([](size_t width, bool is_signed,
                             size_t message_modulus, size_t carry_modulus,
                             size_t degree, size_t lwe_size, size_t n_cts,
                             size_t noise_level, bool ks_first) {
        auto desc = TfhersFheIntDescription();
        desc.width = width;
        desc.is_signed = is_signed;
        desc.message_modulus = message_modulus;
        desc.carry_modulus = carry_modulus;
        desc.degree = degree;
        desc.lwe_size = lwe_size;
        desc.n_cts = n_cts;
        desc.noise_level = noise_level;
        desc.ks_first = ks_first;
        return desc;
      }))
      .def_static("UNKNOWN_NOISE_LEVEL",
                  [] { return concrete_cpu_tfhers_unknown_noise_level(); })
      .def_property(
          "width", [](TfhersFheIntDescription &desc) { return desc.width; },
          [](TfhersFheIntDescription &desc, size_t width) {
            desc.width = width;
          })
      .def_property(
          "message_modulus",
          [](TfhersFheIntDescription &desc) { return desc.message_modulus; },
          [](TfhersFheIntDescription &desc, size_t message_modulus) {
            desc.message_modulus = message_modulus;
          })
      .def_property(
          "carry_modulus",
          [](TfhersFheIntDescription &desc) { return desc.carry_modulus; },
          [](TfhersFheIntDescription &desc, size_t carry_modulus) {
            desc.carry_modulus = carry_modulus;
          })
      .def_property(
          "degree", [](TfhersFheIntDescription &desc) { return desc.degree; },
          [](TfhersFheIntDescription &desc, size_t degree) {
            desc.degree = degree;
          })
      .def_property(
          "lwe_size",
          [](TfhersFheIntDescription &desc) { return desc.lwe_size; },
          [](TfhersFheIntDescription &desc, size_t lwe_size) {
            desc.lwe_size = lwe_size;
          })
      .def_property(
          "n_cts", [](TfhersFheIntDescription &desc) { return desc.n_cts; },
          [](TfhersFheIntDescription &desc, size_t n_cts) {
            desc.n_cts = n_cts;
          })
      .def_property(
          "noise_level",
          [](TfhersFheIntDescription &desc) { return desc.noise_level; },
          [](TfhersFheIntDescription &desc, size_t noise_level) {
            desc.noise_level = noise_level;
          })
      .def_property(
          "is_signed",
          [](TfhersFheIntDescription &desc) { return desc.is_signed; },
          [](TfhersFheIntDescription &desc, bool is_signed) {
            desc.is_signed = is_signed;
          })
      .def_property(
          "ks_first",
          [](TfhersFheIntDescription &desc) { return desc.ks_first; },
          [](TfhersFheIntDescription &desc, bool ks_first) {
            desc.ks_first = ks_first;
          });

  pybind11::enum_<mlir::concretelang::Backend>(m, "Backend")
      .value("CPU", mlir::concretelang::Backend::CPU,
             "Circuit codegen targets cpu.")
      .value("GPU", mlir::concretelang::Backend::GPU,
             "Circuit codegen tartgets gpu.")
      .export_values();

  pybind11::enum_<optimizer::Strategy>(m, "OptimizerStrategy")
      .value("V0", optimizer::Strategy::V0)
      .value("DAG_MONO", optimizer::Strategy::DAG_MONO)
      .value("DAG_MULTI", optimizer::Strategy::DAG_MULTI)
      .export_values();

  pybind11::enum_<concrete_optimizer::MultiParamStrategy>(
      m, "OptimizerMultiParameterStrategy")
      .value("PRECISION", concrete_optimizer::MultiParamStrategy::ByPrecision)
      .value("PRECISION_AND_NORM2",
             concrete_optimizer::MultiParamStrategy::ByPrecisionAndNorm2)
      .export_values();

  pybind11::enum_<concrete_optimizer::Encoding>(m, "Encoding")
      .value("AUTO", concrete_optimizer::Encoding::Auto)
      .value("CRT", concrete_optimizer::Encoding::Crt)
      .value("NATIVE", concrete_optimizer::Encoding::Native)
      .export_values();

  pybind11::class_<CompilationOptions>(m, "CompilationOptions")
      .def(pybind11::init([](mlir::concretelang::Backend backend) {
             return CompilationOptions(backend);
           }),
           arg("backend"))
      .def(
          "set_verify_diagnostics",
          [](CompilationOptions &options, bool b) {
            options.verifyDiagnostics = b;
          },
          "Set option for diagnostics verification.", arg("verify_diagnostics"))
      .def(
          "set_auto_parallelize",
          [](CompilationOptions &options, bool b) {
            options.autoParallelize = b;
          },
          "Set option for auto parallelization.", arg("auto_parallelize"))
      .def(
          "set_loop_parallelize",
          [](CompilationOptions &options, bool b) {
            options.loopParallelize = b;
          },
          "Set option for loop parallelization.", arg("loop_parallelize"))
      .def(
          "set_dataflow_parallelize",
          [](CompilationOptions &options, bool b) {
            options.dataflowParallelize = b;
          },
          "Set option for dataflow parallelization.",
          arg("dataflow_parallelize"))
      .def(
          "set_compress_evaluation_keys",
          [](CompilationOptions &options, bool b) {
            options.compressEvaluationKeys = b;
          },
          "Set option for compression of evaluation keys.",
          arg("compress_evaluation_keys"))
      .def(
          "set_compress_input_ciphertexts",
          [](CompilationOptions &options, bool b) {
            options.compressInputCiphertexts = b;
          },
          "Set option for compression of input ciphertexts.",
          arg("compress_input_ciphertexts"))
      .def(
          "set_optimize_concrete",
          [](CompilationOptions &options, bool b) { options.optimizeTFHE = b; },
          "Set flag to enable/disable optimization of concrete intermediate "
          "representation.",
          arg("optimize"))
      .def(
          "set_p_error",
          [](CompilationOptions &options, double p_error) {
            options.optimizerConfig.p_error = p_error;
          },
          "Set error probability for shared by each pbs.", arg("p_error"))
      .def(
          "set_display_optimizer_choice",
          [](CompilationOptions &options, bool display) {
            options.optimizerConfig.display = display;
          },
          "Set display flag of optimizer choices.", arg("display"))
      .def(
          "set_optimizer_strategy",
          [](CompilationOptions &options, optimizer::Strategy strategy) {
            options.optimizerConfig.strategy = strategy;
          },
          "Set the strategy of the optimizer.", arg("strategy"))
      .def(
          "set_optimizer_multi_parameter_strategy",
          [](CompilationOptions &options,
             concrete_optimizer::MultiParamStrategy strategy) {
            options.optimizerConfig.multi_param_strategy = strategy;
          },
          "Set the strategy of the optimizer for multi-parameter.",
          arg("strategy"))
      .def(
          "set_global_p_error",
          [](CompilationOptions &options, double global_p_error) {
            options.optimizerConfig.global_p_error = global_p_error;
          },
          "Set global error probability for the full circuit.",
          arg("global_p_error"))
      .def(
          "add_composition",
          [](CompilationOptions &options, std::string from_func,
             size_t from_pos, std::string to_func, size_t to_pos) {
            options.optimizerConfig.composition_rules.push_back(
                {from_func, from_pos, to_func, to_pos});
          },
          "Add a composition rule.", arg("from_func"), arg("from_pos"),
          arg("to_func"), arg("to_pos"))
      .def(
          "set_composable",
          [](CompilationOptions &options, bool composable) {
            options.optimizerConfig.composable = composable;
          },
          "Set composable flag.", arg("composable"))
      .def(
          "set_security_level",
          [](CompilationOptions &options, int security_level) {
            options.optimizerConfig.security = security_level;
          },
          "Set security level.", arg("security_level"))
      .def(
          "set_v0_parameter",
          [](CompilationOptions &options, size_t glweDimension,
             size_t logPolynomialSize, size_t nSmall, size_t brLevel,
             size_t brLogBase, size_t ksLevel, size_t ksLogBase) {
            options.v0Parameter = {glweDimension, logPolynomialSize, nSmall,
                                   brLevel,       brLogBase,         ksLevel,
                                   ksLogBase,     std::nullopt};
          },
          "Set the basic V0 parameters.", arg("glwe_dimension"),
          arg("log_poly_size"), arg("n_small"), arg("br_level"),
          arg("br_log_base"), arg("ks_level"), arg("ks_log_base"))
      .def(
          "set_all_v0_parameter",
          [](CompilationOptions &options, size_t glweDimension,
             size_t logPolynomialSize, size_t nSmall, size_t brLevel,
             size_t brLogBase, size_t ksLevel, size_t ksLogBase,
             std::vector<int64_t> crtDecomposition, size_t cbsLevel,
             size_t cbsLogBase, size_t pksLevel, size_t pksLogBase,
             size_t pksInputLweDimension, size_t pksOutputPolynomialSize) {
            mlir::concretelang::PackingKeySwitchParameter pksParam = {
                pksInputLweDimension, pksOutputPolynomialSize, pksLevel,
                pksLogBase};
            mlir::concretelang::CitcuitBoostrapParameter crbParam = {
                cbsLevel, cbsLogBase};
            mlir::concretelang::WopPBSParameter wopPBSParam = {pksParam,
                                                               crbParam};
            mlir::concretelang::LargeIntegerParameter largeIntegerParam = {
                crtDecomposition, wopPBSParam};
            options.v0Parameter = {glweDimension, logPolynomialSize, nSmall,
                                   brLevel,       brLogBase,         ksLevel,
                                   ksLogBase,     largeIntegerParam};
          },
          "Set all the V0 parameters.", arg("glwe_dimension"),
          arg("log_poly_size"), arg("n_small"), arg("br_level"),
          arg("br_log_base"), arg("ks_level"), arg("ks_log_base"),
          arg("crt_decomp"), arg("cbs_level"), arg("cbs_log_base"),
          arg("pks_level"), arg("pks_log_base"), arg("pks_input_lwe_dim"),
          arg("pks_output_poly_size"))
      .def(
          "force_encoding",
          [](CompilationOptions &options,
             concrete_optimizer::Encoding encoding) {
            options.optimizerConfig.encoding = encoding;
          },
          "Force the compiler to use a specific encoding.", arg("encoding"))
      .def(
          "simulation",
          [](CompilationOptions &options, bool simulate) {
            options.simulate = simulate;
          },
          "Enable or disable simulation.", arg("simulate"))
      .def(
          "set_emit_gpu_ops",
          [](CompilationOptions &options, bool emit_gpu_ops) {
            options.emitGPUOps = emit_gpu_ops;
          },
          "Set flag that allows gpu ops to be emitted.", arg("emit_gpu_ops"))
      .def(
          "set_batch_tfhe_ops",
          [](CompilationOptions &options, bool batch_tfhe_ops) {
            options.batchTFHEOps = batch_tfhe_ops;
          },
          "Set flag that triggers the batching of scalar TFHE operations.",
          arg("batch_tfhe_ops"))
      .def(
          "set_enable_tlu_fusing",
          [](CompilationOptions &options, bool enableTluFusing) {
            options.enableTluFusing = enableTluFusing;
          },
          "Enable or disable tlu fusing.", arg("enable_tlu_fusing"))
      .def(
          "set_print_tlu_fusing",
          [](CompilationOptions &options, bool printTluFusing) {
            options.printTluFusing = printTluFusing;
          },
          "Enable or disable printing tlu fusing.", arg("print_tlu_fusing"))
      .def(
          "set_enable_overflow_detection_in_simulation",
          [](CompilationOptions &options, bool enableOverflowDetection) {
            options.enableOverflowDetectionInSimulation =
                enableOverflowDetection;
          },
          "Enable or disable overflow detection during simulation.",
          arg("enable_overflow_detection"))
      .doc() = "Holds different flags and options of the compilation process.";

  pybind11::enum_<mlir::concretelang::PrimitiveOperation>(m,
                                                          "PrimitiveOperation")
      .value("PBS", mlir::concretelang::PrimitiveOperation::PBS)
      .value("WOP_PBS", mlir::concretelang::PrimitiveOperation::WOP_PBS)
      .value("KEY_SWITCH", mlir::concretelang::PrimitiveOperation::KEY_SWITCH)
      .value("CLEAR_ADDITION",
             mlir::concretelang::PrimitiveOperation::CLEAR_ADDITION)
      .value("ENCRYPTED_ADDITION",
             mlir::concretelang::PrimitiveOperation::ENCRYPTED_ADDITION)
      .value("CLEAR_MULTIPLICATION",
             mlir::concretelang::PrimitiveOperation::CLEAR_MULTIPLICATION)
      .value("ENCRYPTED_NEGATION",
             mlir::concretelang::PrimitiveOperation::ENCRYPTED_NEGATION)
      .export_values();

  pybind11::enum_<mlir::concretelang::KeyType>(m, "KeyType")
      .value("SECRET", mlir::concretelang::KeyType::SECRET)
      .value("BOOTSTRAP", mlir::concretelang::KeyType::BOOTSTRAP)
      .value("KEY_SWITCH", mlir::concretelang::KeyType::KEY_SWITCH)
      .value("PACKING_KEY_SWITCH",
             mlir::concretelang::KeyType::PACKING_KEY_SWITCH)
      .export_values();

  pybind11::class_<mlir::concretelang::Statistic>(m, "Statistic")
      .def_readonly("operation", &mlir::concretelang::Statistic::operation)
      .def_readonly("location", &mlir::concretelang::Statistic::location)
      .def_readonly("keys", &mlir::concretelang::Statistic::keys)
      .def_readonly("count", &mlir::concretelang::Statistic::count);

  pybind11::class_<mlir::concretelang::CircuitCompilationFeedback>(
      m, "CircuitCompilationFeedback")
      .def_readonly("name",
                    &mlir::concretelang::CircuitCompilationFeedback::name)
      .def_readonly(
          "total_inputs_size",
          &mlir::concretelang::CircuitCompilationFeedback::totalInputsSize)
      .def_readonly(
          "total_output_size",
          &mlir::concretelang::CircuitCompilationFeedback::totalOutputsSize)
      .def_readonly("crt_decompositions_of_outputs",
                    &mlir::concretelang::CircuitCompilationFeedback::
                        crtDecompositionsOfOutputs)
      .def_readonly("statistics",
                    &mlir::concretelang::CircuitCompilationFeedback::statistics)
      .def_readonly(
          "memory_usage_per_location",
          &mlir::concretelang::CircuitCompilationFeedback::memoryUsagePerLoc)
      .doc() = "Compilation feedback for a single circuit.";

  pybind11::class_<mlir::concretelang::ProgramCompilationFeedback>(
      m, "ProgramCompilationFeedback")
      .def_readonly("complexity",
                    &mlir::concretelang::ProgramCompilationFeedback::complexity)
      .def_readonly("p_error",
                    &mlir::concretelang::ProgramCompilationFeedback::pError)
      .def_readonly(
          "global_p_error",
          &mlir::concretelang::ProgramCompilationFeedback::globalPError)
      .def_readonly(
          "total_secret_keys_size",
          &mlir::concretelang::ProgramCompilationFeedback::totalSecretKeysSize)
      .def_readonly("total_bootstrap_keys_size",
                    &mlir::concretelang::ProgramCompilationFeedback::
                        totalBootstrapKeysSize)
      .def_readonly("total_keyswitch_keys_size",
                    &mlir::concretelang::ProgramCompilationFeedback::
                        totalKeyswitchKeysSize)
      .def_readonly(
          "circuit_feedbacks",
          &mlir::concretelang::ProgramCompilationFeedback::circuitFeedbacks)
      .doc() = "Compilation feedback for a whole program.";

  pybind11::class_<mlir::concretelang::CompilationContext,
                   std::shared_ptr<mlir::concretelang::CompilationContext>>(
      m, "CompilationContext")
      .def(pybind11::init([]() {
        return mlir::concretelang::CompilationContext::createShared();
      }))
      .def("mlir_context",
           [](std::shared_ptr<mlir::concretelang::CompilationContext> cctx) {
             auto mlirCtx = cctx->getMLIRContext();
             return pybind11::reinterpret_steal<pybind11::object>(
                 mlirPythonContextToCapsule(wrap(mlirCtx)));
           });

  // ------------------------------------------------------------------------------//
  // LWE SECRET KEY PARAM //
  // ------------------------------------------------------------------------------//

  struct LweSecretKeyParam {
    Message<concreteprotocol::LweSecretKeyInfo> info;
  };
  pybind11::class_<LweSecretKeyParam>(m, "LweSecretKeyParam")
      .def(init([]() -> LweSecretKeyParam {
        throw std::runtime_error("Explicit construction forbidden.");
      }))
      .def(
          "dimension",
          [](LweSecretKeyParam &key) {
            return key.info.asReader().getParams().getLweDimension();
          },
          "Return the associated LWE dimension.")
      .doc() = "Parameters of an LWE Secret Key.";

  // ------------------------------------------------------------------------------//
  // BOOTSTRAP KEY PARAM //
  // ------------------------------------------------------------------------------//

  struct BootstrapKeyParam {
    Message<concreteprotocol::LweBootstrapKeyInfo> info;
  };
  pybind11::class_<BootstrapKeyParam>(m, "BootstrapKeyParam")
      .def(init([]() -> BootstrapKeyParam {
        throw std::runtime_error("Explicit construction forbidden.");
      }))
      .def(
          "input_secret_key_id",
          [](BootstrapKeyParam &key) {
            return key.info.asReader().getInputId();
          },
          "Return the key id of the associated input key.")
      .def(
          "output_secret_key_id",
          [](BootstrapKeyParam &key) {
            return key.info.asReader().getOutputId();
          },
          "Return the key id of the associated output key.")
      .def(
          "level",
          [](BootstrapKeyParam &key) {
            return key.info.asReader().getParams().getLevelCount();
          },
          "Return the associated number of levels.")
      .def(
          "base_log",
          [](BootstrapKeyParam &key) {
            return key.info.asReader().getParams().getBaseLog();
          },
          "Return the associated base log.")
      .def(
          "glwe_dimension",
          [](BootstrapKeyParam &key) {
            return key.info.asReader().getParams().getGlweDimension();
          },
          "Return the associated GLWE dimension.")
      .def(
          "variance",
          [](BootstrapKeyParam &key) {
            return key.info.asReader().getParams().getVariance();
          },
          "Return the associated noise variance.")
      .def(
          "polynomial_size",
          [](BootstrapKeyParam &key) {
            return key.info.asReader().getParams().getPolynomialSize();
          },
          "Return the associated polynomial size.")
      .def(
          "input_lwe_dimension",
          [](BootstrapKeyParam &key) {
            return key.info.asReader().getParams().getInputLweDimension();
          },
          "Return the associated input lwe dimension.")
      .doc() = "Parameters of a Bootstrap key.";

  // ------------------------------------------------------------------------------//
  // KEYSWITCH KEY PARAM //
  // ------------------------------------------------------------------------------//

  struct KeyswitchKeyParam {
    Message<concreteprotocol::LweKeyswitchKeyInfo> info;
  };
  pybind11::class_<KeyswitchKeyParam>(m, "KeyswitchKeyParam")
      .def(init([]() -> KeyswitchKeyParam {
        throw std::runtime_error("Explicit construction forbidden.");
      }))
      .def(
          "input_secret_key_id",
          [](KeyswitchKeyParam &key) {
            return key.info.asReader().getInputId();
          },
          "Return the key id of the associated input key.")
      .def(
          "output_secret_key_id",
          [](KeyswitchKeyParam &key) {
            return key.info.asReader().getOutputId();
          },
          "Return the key id of the associated output key.")
      .def(
          "level",
          [](KeyswitchKeyParam &key) {
            return key.info.asReader().getParams().getLevelCount();
          },
          "Return the associated number of levels.")
      .def(
          "base_log",
          [](KeyswitchKeyParam &key) {
            return key.info.asReader().getParams().getBaseLog();
          },
          "Return the associated base log.")
      .def(
          "variance",
          [](KeyswitchKeyParam &key) {
            return key.info.asReader().getParams().getVariance();
          },
          "Return the associated noise variance.")
      .doc() = "Parameters of a keyswitch key.";

  // ------------------------------------------------------------------------------//
  // PACKING KEYSWITCH KEY PARAM //
  // ------------------------------------------------------------------------------//

  struct PackingKeyswitchKeyParam {
    Message<concreteprotocol::PackingKeyswitchKeyInfo> info;
  };
  pybind11::class_<PackingKeyswitchKeyParam>(m, "PackingKeyswitchKeyParam")
      .def(init([]() -> PackingKeyswitchKeyParam {
        throw std::runtime_error("Explicit construction forbidden.");
      }))
      .def(
          "input_secret_key_id",
          [](PackingKeyswitchKeyParam &key) {
            return key.info.asReader().getInputId();
          },
          "Return the key id of the associated input key.")
      .def(
          "output_secret_key_id",
          [](PackingKeyswitchKeyParam &key) {
            return key.info.asReader().getOutputId();
          },
          "Return the key id of the associated output key.")
      .def(
          "level",
          [](PackingKeyswitchKeyParam &key) {
            return key.info.asReader().getParams().getLevelCount();
          },
          "Return the associated number of levels.")
      .def(
          "base_log",
          [](PackingKeyswitchKeyParam &key) {
            return key.info.asReader().getParams().getBaseLog();
          },
          "Return the associated base log.")
      .def(
          "glwe_dimension",
          [](PackingKeyswitchKeyParam &key) {
            return key.info.asReader().getParams().getGlweDimension();
          },
          "Return the associated GLWE dimension.")
      .def(
          "polynomial_size",
          [](PackingKeyswitchKeyParam &key) {
            return key.info.asReader().getParams().getPolynomialSize();
          },
          "Return the associated polynomial size.")
      .def(
          "input_lwe_dimension",
          [](PackingKeyswitchKeyParam &key) {
            return key.info.asReader().getParams().getInputLweDimension();
          },
          "Return the associated input LWE dimension.")
      .def(
          "variance",
          [](PackingKeyswitchKeyParam &key) {
            return key.info.asReader().getParams().getVariance();
          },
          "Return the associated noise variance.")
      .doc() = "Parameters of a packing keyswitch key.";

  // ------------------------------------------------------------------------------//
  // PROGRAM INFO //
  // ------------------------------------------------------------------------------//

  struct ProgramInfo {
    Message<concreteprotocol::ProgramInfo> programInfo;

    concreteprotocol::LweCiphertextEncryptionInfo::Reader
    inputEncryptionAt(size_t inputId, std::string circuitName) {
      auto reader = programInfo.asReader();
      if (!reader.hasCircuits()) {
        throw std::runtime_error("can't get keyid: no circuit info");
      }
      auto circuits = reader.getCircuits();
      for (auto circuit : circuits) {
        if (circuit.hasName() &&
            circuitName.compare(circuit.getName().cStr()) == 0) {
          if (!circuit.hasInputs()) {
            throw std::runtime_error("can't get keyid: no input");
          }
          auto inputs = circuit.getInputs();
          if (inputId >= inputs.size()) {
            throw std::runtime_error(
                "can't get keyid: inputId bigger than number of inputs");
          }
          auto input = inputs[inputId];
          if (!input.hasTypeInfo()) {
            throw std::runtime_error(
                "can't get keyid: input don't have typeInfo");
          }
          auto typeInfo = input.getTypeInfo();
          if (!typeInfo.hasLweCiphertext()) {
            throw std::runtime_error("can't get keyid: typeInfo don't "
                                     "have lwe ciphertext info");
          }
          auto lweCt = typeInfo.getLweCiphertext();
          if (!lweCt.hasEncryption()) {
            throw std::runtime_error("can't get keyid: lwe ciphertext "
                                     "don't have encryption info");
          }
          return lweCt.getEncryption();
        }
      }

      throw std::runtime_error("can't get keyid: no circuit with name " +
                               circuitName);
    }
  };
  pybind11::class_<ProgramInfo>(m, "ProgramInfo")
      .def(init([]() -> ProgramInfo {
        throw std::runtime_error("Explicit construction forbidden.");
      }))
      .def_static(
          "deserialize",
          [](const pybind11::bytes &buffer) {
            auto programInfo = Message<concreteprotocol::ProgramInfo>();
            if (programInfo.readJsonFromString(buffer).has_failure()) {
              throw std::runtime_error("Failed to deserialize program info");
            }
            return ProgramInfo{programInfo};
          },
          "Deserialize a ProgramInfo from bytes.", arg("bytes"))
      .def(
          "serialize",
          [](ProgramInfo &programInfo) {
            auto programInfoSerialize = [](ProgramInfo &params) {
              auto maybeJson = params.programInfo.writeJsonToString();
              if (maybeJson.has_failure()) {
                throw std::runtime_error("Failed to serialize program info");
              }
              return maybeJson.value();
            };
            return pybind11::bytes(programInfoSerialize(programInfo));
          },
          "Serialize a ProgramInfo to bytes.")
      .def(
          "input_keyid_at",
          [](ProgramInfo &programInfo, size_t pos, std::string circuitName) {
            auto encryption = programInfo.inputEncryptionAt(pos, circuitName);
            return encryption.getKeyId();
          },
          "Return the key id associated to the argument `pos` of circuit "
          "`circuit_name`.",
          arg("pos"), arg("circuit_name"))
      .def(
          "input_variance_at",
          [](ProgramInfo &programInfo, size_t pos, std::string circuitName) {
            auto encryption = programInfo.inputEncryptionAt(pos, circuitName);
            return encryption.getVariance();
          },
          "Return the noise variance associated to the argument `pos` of "
          "circuit `circuit_name`.",
          arg("pos"), arg("circuit_name"))
      .def("function_list",
           [](ProgramInfo &programInfo) {
             std::vector<std::string> result;
             for (auto circuit :
                  programInfo.programInfo.asReader().getCircuits()) {
               result.push_back(circuit.getName());
             }
             return result;
           })
      .def(
          "output_signs",
          [](ProgramInfo &programInfo) {
            std::vector<bool> result;
            for (auto output : programInfo.programInfo.asReader()
                                   .getCircuits()[0]
                                   .getOutputs()) {
              if (output.getTypeInfo().hasLweCiphertext() &&
                  output.getTypeInfo()
                      .getLweCiphertext()
                      .getEncoding()
                      .hasInteger()) {
                result.push_back(output.getTypeInfo()
                                     .getLweCiphertext()
                                     .getEncoding()
                                     .getInteger()
                                     .getIsSigned());
              } else {
                result.push_back(true);
              }
            }
            return result;
          },
          "Return the signedness of the output of the first circuit.")
      .def(
          "input_signs",
          [](ProgramInfo &programInfo) {
            std::vector<bool> result;
            for (auto input : programInfo.programInfo.asReader()
                                  .getCircuits()[0]
                                  .getInputs()) {
              if (input.getTypeInfo().hasLweCiphertext() &&
                  input.getTypeInfo()
                      .getLweCiphertext()
                      .getEncoding()
                      .hasInteger()) {
                result.push_back(input.getTypeInfo()
                                     .getLweCiphertext()
                                     .getEncoding()
                                     .getInteger()
                                     .getIsSigned());
              } else {
                result.push_back(true);
              }
            }
            return result;
          },
          "Return the signedness of the input of the first circuit.")
      .def(
          "secret_keys",
          [](ProgramInfo &programInfo) {
            auto secretKeys = std::vector<LweSecretKeyParam>();
            for (auto key : programInfo.programInfo.asReader()
                                .getKeyset()
                                .getLweSecretKeys()) {
              secretKeys.push_back(LweSecretKeyParam{key});
            }
            return secretKeys;
          },
          "Return the parameters of the secret keys for this program.")
      .def(
          "bootstrap_keys",
          [](ProgramInfo &programInfo) {
            auto bootstrapKeys = std::vector<BootstrapKeyParam>();
            for (auto key : programInfo.programInfo.asReader()
                                .getKeyset()
                                .getLweBootstrapKeys()) {
              bootstrapKeys.push_back(BootstrapKeyParam{key});
            }
            return bootstrapKeys;
          },
          "Return the parameters of the bootstrap keys for this program.")
      .def(
          "keyswitch_keys",
          [](ProgramInfo &programInfo) {
            auto keyswitchKeys = std::vector<KeyswitchKeyParam>();
            for (auto key : programInfo.programInfo.asReader()
                                .getKeyset()
                                .getLweKeyswitchKeys()) {
              keyswitchKeys.push_back(KeyswitchKeyParam{key});
            }
            return keyswitchKeys;
          },
          "Return the parameters of the keyswitch keys for this program.")
      .def(
          "packing_keyswitch_keys",
          [](ProgramInfo &programInfo) {
            auto packingKeyswitchKeys = std::vector<PackingKeyswitchKeyParam>();
            for (auto key : programInfo.programInfo.asReader()
                                .getKeyset()
                                .getPackingKeyswitchKeys()) {
              packingKeyswitchKeys.push_back(PackingKeyswitchKeyParam{key});
            }
            return packingKeyswitchKeys;
          },
          "Return the parameters of the packing keyswitch keys for this "
          "program.")
      .doc() = "Informations describing a compiled program.";

  // ------------------------------------------------------------------------------//
  // KEYSET CACHE //
  // ------------------------------------------------------------------------------//

  pybind11::class_<KeysetCache>(m, "KeysetCache")
      .def(pybind11::init<std::string &>(), arg("backing_directory_path"))
      .doc() = "Local keysets cache.";

  // ------------------------------------------------------------------------------//
  // SERVER KEYSET //
  // ------------------------------------------------------------------------------//
  pybind11::class_<ServerKeyset>(m, "ServerKeyset")
      .def(init([]() -> ServerKeyset {
        throw std::runtime_error("Explicit construction forbidden.");
      }))
      .def_static(
          "deserialize",
          [](const pybind11::bytes &buffer) {
            auto serverKeysetProto = Message<concreteprotocol::ServerKeyset>();
            auto maybeError = serverKeysetProto.readBinaryFromString(
                buffer, mlir::concretelang::python::DESER_OPTIONS);
            if (maybeError.has_failure()) {
              throw std::runtime_error("Failed to deserialize server keyset." +
                                       maybeError.as_failure().error().mesg);
            }
            return ServerKeyset::fromProto(serverKeysetProto);
          },
          "Deserialize a ServerKeyset from bytes.", arg("bytes"))
      .def(
          "serialize",
          [](ServerKeyset &serverKeyset) {
            auto serverKeysetSerialize = [](ServerKeyset &serverKeyset) {
              auto serverKeysetProto = serverKeyset.toProto();
              auto maybeBuffer = serverKeysetProto.writeBinaryToString();
              if (maybeBuffer.has_failure()) {
                throw std::runtime_error("Failed to serialize server keyset.");
              }
              return maybeBuffer.value();
            };
            return pybind11::bytes(serverKeysetSerialize(serverKeyset));
          },
          "Serialize a ServerKeyset to bytes.")
      .doc() = "Server-side / Evaluation keyset";

  // ------------------------------------------------------------------------------//
  // KEYSET //
  // ------------------------------------------------------------------------------//

  pybind11::class_<Keyset>(m, "Keyset")
      .def(init([](ProgramInfo programInfo, std::optional<KeysetCache> cache,
                   uint64_t secretSeedMsb, uint64_t secretSeedLsb,
                   uint64_t encSeedMsb, uint64_t encSeedLsb) {
             SignalGuard const signalGuard;

             auto secretSeed =
                 (((__uint128_t)secretSeedMsb) << 64) | secretSeedLsb;
             auto encryptionSeed =
                 (((__uint128_t)encSeedMsb) << 64) | encSeedLsb;

             if (cache) {
               GET_OR_THROW_RESULT(
                   Keyset keyset,
                   (*cache).getKeyset(
                       programInfo.programInfo.asReader().getKeyset(),
                       secretSeed, encryptionSeed));
               return std::make_unique<Keyset>(std::move(keyset));
             } else {
               ::concretelang::csprng::SecretCSPRNG secCsprng(secretSeed);
               ::concretelang::csprng::EncryptionCSPRNG encCsprng(
                   encryptionSeed);
               auto keyset =
                   Keyset(programInfo.programInfo.asReader().getKeyset(),
                          secCsprng, encCsprng);
               return std::make_unique<Keyset>(std::move(keyset));
             }
           }),
           arg("program_info"), arg("keyset_cache"), arg("secret_seed_msb") = 0,
           arg("secret_seed_lsb") = 0, arg("encryption_seed_msb") = 0,
           arg("encryption_seed_lsb") = 0)
      .def_static(
          "deserialize",
          [](const pybind11::bytes &buffer) {
            auto keysetProto = Message<concreteprotocol::Keyset>();
            auto maybeError = keysetProto.readBinaryFromString(
                buffer, mlir::concretelang::python::DESER_OPTIONS);
            if (maybeError.has_failure()) {
              throw std::runtime_error("Failed to deserialize keyset." +
                                       maybeError.as_failure().error().mesg);
            }
            auto keyset = Keyset::fromProto(keysetProto);
            return std::make_unique<Keyset>(std::move(keyset));
          },
          "Deserialize a Keyset from bytes.", arg("bytes"))
      .def(
          "serialize",
          [](Keyset &keySet) {
            auto keySetSerialize = [](Keyset &keyset) {
              auto keysetProto = keyset.toProto();
              auto maybeBuffer = keysetProto.writeBinaryToString();
              if (maybeBuffer.has_failure()) {
                throw std::runtime_error("Failed to serialize keys.");
              }
              return maybeBuffer.value();
            };
            return pybind11::bytes(keySetSerialize(keySet));
          },
          "Serialize a Keyset to bytes.")
      .def(
          "serialize_lwe_secret_key_as_glwe",
          [](Keyset &keyset, size_t keyIndex, size_t glwe_dimension,
             size_t polynomial_size) {
            auto secretKeys = keyset.client.lweSecretKeys;
            if (keyIndex >= secretKeys.size()) {
              throw std::runtime_error(
                  "keyIndex is bigger than the number of keys");
            }
            auto secretKey = secretKeys[keyIndex];
            auto skBuffer = secretKey.getBuffer();
            auto buffer_size = concrete_cpu_glwe_secret_key_buffer_size_u64(
                glwe_dimension, polynomial_size);
            std::vector<uint8_t> buffer(buffer_size, 0);
            buffer_size = concrete_cpu_serialize_glwe_secret_key_u64(
                skBuffer.data(), glwe_dimension, polynomial_size, buffer.data(),
                buffer_size);
            if (buffer_size == 0) {
              throw std::runtime_error("couldn't serialize the secret key");
            }
            auto bytes = pybind11::bytes((char *)buffer.data(), buffer_size);
            return bytes;
          },
          "Serialize the `key_id` secret key as a tfhe-rs GLWE key with "
          "parameters `glwe_dim` and `poly_size`.",
          arg("key_id"), arg("glwe_dim"), arg("poly_size"))
      .def(
          "get_server_keys",
          [](Keyset &keyset) { return ServerKeyset{keyset.server}; },
          "Return the associated ServerKeyset.")
      .doc() =
      "Complete keyset containing both client-side and server-side keys.";

  // ------------------------------------------------------------------------------//
  // LIBRARY //
  // ------------------------------------------------------------------------------//

  pybind11::class_<Library>(m, "Library")
      .def(init([](std::string output_dir_path) -> Library {
             return Library(output_dir_path);
           }),
           arg("output_dir_path"))
      .def(
          "get_program_info",
          [](Library &library) {
            return ProgramInfo{library.getProgramInfo()};
          },
          "Return the program info associated to the library.")
      .def(
          "get_shared_lib_path",
          [](Library &library) { return library.getOutputDirPath(); },
          "Return the path to the shared library.")
      .def(
          "get_program_info_path",
          [](Library &library) { return library.getProgramInfoPath(); },
          "Return the path to the program info file.")
      .def(
          "get_program_compilation_feedback",
          [](Library &library) {
            auto path = library.getCompilationFeedbackPath();
            GET_OR_THROW_RESULT(auto feedback,
                                ProgramCompilationFeedback::load(path));
            return feedback;
          },
          "Return the associated program compilation feedback.")
      .doc() = "Library object representing the output of a compilation.";

  // ------------------------------------------------------------------------------//
  // COMPILER //
  // ------------------------------------------------------------------------------//

  struct Compiler {
    std::string outputPath;
    std::string runtimeLibraryPath;
    bool generateSharedLib;
    bool generateStaticLib;
    bool generateClientParameters;
    bool generateCompilationFeedback;
  };
  pybind11::class_<Compiler>(m, "Compiler")
      .def(init([](std::string outputPath, std::string runtimeLibraryPath,
                   bool generateSharedLib, bool generateStaticLib,
                   bool generateProgramInfo, bool generateCompilationFeedback) {
             return Compiler{outputPath.c_str(),  runtimeLibraryPath.c_str(),
                             generateSharedLib,   generateStaticLib,
                             generateProgramInfo, generateCompilationFeedback};
           }),
           arg("output_path"), arg("runtime_lib_path"),
           arg("generate_shared_lib") = false,
           arg("generate_static_lib") = false,
           arg("generate_program_info") = false,
           arg("generate_compilation_feedback") = false)
      .def(
          "compile",
          [](Compiler &support, std::string mlir_program,
             mlir::concretelang::CompilationOptions options) {
            SignalGuard signalGuard;
            llvm::SourceMgr sm;
            sm.AddNewSourceBuffer(
                llvm::MemoryBuffer::getMemBuffer(mlir_program.c_str()),
                llvm::SMLoc());

            // Setup the compiler engine
            auto context = CompilationContext::createShared();
            concretelang::CompilerEngine engine(context);
            engine.setCompilationOptions(options);

            // Compile to a library
            GET_OR_THROW_EXPECTED(
                auto library,
                engine.compile(
                    sm, support.outputPath, support.runtimeLibraryPath,
                    support.generateSharedLib, support.generateStaticLib,
                    support.generateClientParameters,
                    support.generateCompilationFeedback));
            return library;
          },
          "Compile `mlir_program` using the `options` compilation options.",
          arg("mlir_program"), arg("options"))
      .def(
          "compile",
          [](Compiler &support, pybind11::object mlir_module,
             mlir::concretelang::CompilationOptions options,
             std::shared_ptr<mlir::concretelang::CompilationContext> context) {
            SignalGuard signalGuard;
            mlir::ModuleOp module =
                unwrap(mlirPythonCapsuleToModule(mlir_module.ptr())).clone();

            // Setup the compiler engine
            concretelang::CompilerEngine engine(context);
            engine.setCompilationOptions(options);

            // Compile to a library
            GET_OR_THROW_EXPECTED(
                auto library,
                engine.compile(
                    module, support.outputPath, support.runtimeLibraryPath,
                    support.generateSharedLib, support.generateStaticLib,
                    support.generateClientParameters,
                    support.generateCompilationFeedback));
            return library;
          },
          "Compile the `mlir_module` module with `options` compilation "
          "options, under the `context` compilation context.",
          arg("mlir_module"), arg("options"), arg("context").none(false))
      .doc() = "Provides compilation facility.";

  // ------------------------------------------------------------------------------//
  // TRANSPORT VALUE //
  // ------------------------------------------------------------------------------//

  pybind11::class_<TransportValue>(m, "TransportValue")
      .def_static(
          "deserialize",
          [](const pybind11::bytes &buffer) {
            auto inner = TransportValue();
            if (inner
                    .readBinaryFromString(
                        buffer, mlir::concretelang::python::DESER_OPTIONS)
                    .has_failure()) {
              throw std::runtime_error("Failed to deserialize TransportValue");
            }
            return TransportValue{inner};
          },
          "Deserialize a TransportValue from bytes.", arg("bytes"))
      .def(
          "serialize",
          [](const TransportValue &value) {
            auto valueSerialize = [](const TransportValue &value) {
              auto maybeString = value.writeBinaryToString();
              if (maybeString.has_failure()) {
                throw std::runtime_error("Failed to serialize TransportValue");
              }
              return maybeString.value();
            };
            return pybind11::bytes(valueSerialize(value));
          },
          "Serialize a TransportValue to bytes")
      .doc() = "Public/Transportable value.";

  // ------------------------------------------------------------------------------//
  // VALUE //
  // ------------------------------------------------------------------------------//

  typedef std::variant<int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                       uint32_t, uint64_t, array>
      PyValType;

  pybind11::class_<Value>(m, "Value")
      .def(init([](int64_t scalar) { return Value{Tensor<int64_t>(scalar)}; }),
           arg("input"))
      .def(init([](array input) -> Value {
             if (input.dtype().kind() == 'i') {
               if (input.dtype().itemsize() == 1) {
                 return Value{::arrayToTensor<int8_t>(input)};
               }
               if (input.dtype().itemsize() == 2) {
                 return Value{::arrayToTensor<int16_t>(input)};
               }
               if (input.dtype().itemsize() == 4) {
                 return Value{::arrayToTensor<int32_t>(input)};
               }
               if (input.dtype().itemsize() == 8) {
                 return Value{::arrayToTensor<int64_t>(input)};
               }
             }
             if (input.dtype().kind() == 'u') {
               if (input.dtype().itemsize() == 1) {
                 return Value{::arrayToTensor<uint8_t>(input)};
               }
               if (input.dtype().itemsize() == 2) {
                 return Value{::arrayToTensor<uint16_t>(input)};
               }
               if (input.dtype().itemsize() == 4) {
                 return Value{::arrayToTensor<uint32_t>(input)};
               }
               if (input.dtype().itemsize() == 8) {
                 return Value{::arrayToTensor<uint64_t>(input)};
               }
             }
             throw std::runtime_error(
                 "Values can only be constructed from arrays "
                 "of signed and unsigned integers.");
           }),
           arg("input"))
      .def(init([](pybind11::object object) -> Value {
             throw std::runtime_error("Failed to create value from input.");
           }),
           arg("input"))
      .def(
          "to_py_val",
          [](Value &value) -> PyValType {
            if (value.isScalar()) {
              if (value.hasElementType<int8_t>()) {
                return {value.getTensor<int8_t>()->values[0]};
              }
              if (value.hasElementType<int16_t>()) {
                return {value.getTensor<int16_t>()->values[0]};
              }
              if (value.hasElementType<int32_t>()) {
                return {value.getTensor<int32_t>()->values[0]};
              }
              if (value.hasElementType<int64_t>()) {
                return {value.getTensor<int64_t>()->values[0]};
              }
              if (value.hasElementType<uint8_t>()) {
                return {value.getTensor<uint8_t>()->values[0]};
              }
              if (value.hasElementType<uint16_t>()) {
                return {value.getTensor<uint16_t>()->values[0]};
              }
              if (value.hasElementType<uint32_t>()) {
                return {value.getTensor<uint32_t>()->values[0]};
              }
              if (value.hasElementType<uint64_t>()) {
                return {value.getTensor<uint64_t>()->values[0]};
              }
            } else {
              if (value.hasElementType<int8_t>()) {
                return {tensorToArray(value.getTensor<int8_t>().value())};
              }
              if (value.hasElementType<int16_t>()) {
                return {tensorToArray(value.getTensor<int16_t>().value())};
              }
              if (value.hasElementType<int32_t>()) {
                return {tensorToArray(value.getTensor<int32_t>().value())};
              }
              if (value.hasElementType<int64_t>()) {
                return {tensorToArray(value.getTensor<int64_t>().value())};
              }
              if (value.hasElementType<uint8_t>()) {
                return {tensorToArray(value.getTensor<uint8_t>().value())};
              }
              if (value.hasElementType<uint16_t>()) {
                return {tensorToArray(value.getTensor<uint16_t>().value())};
              }
              if (value.hasElementType<uint32_t>()) {
                return {tensorToArray(value.getTensor<uint32_t>().value())};
              }
              if (value.hasElementType<uint64_t>()) {
                return {tensorToArray(value.getTensor<uint64_t>().value())};
              }
            }
            throw std::invalid_argument("Value has insupported scalar type.");
          },
          "Return the inner value as a python type.")
      .def(
          "is_tensor", [](Value &value) { return !value.isScalar(); },
          "Return if the value is a tensor (as opposed to a scalar).")
      .def(
          "get_unsigned_tensor_data",
          [](Value &value) {
            if (auto tensor = value.getTensor<uint8_t>(); tensor) {
              Tensor<uint64_t> out = (Tensor<uint64_t>)tensor.value();
              return out.values;
            } else if (auto tensor = value.getTensor<uint16_t>(); tensor) {
              Tensor<uint64_t> out = (Tensor<uint64_t>)tensor.value();
              return out.values;
            } else if (auto tensor = value.getTensor<uint32_t>(); tensor) {
              Tensor<uint64_t> out = (Tensor<uint64_t>)tensor.value();
              return out.values;
            } else if (auto tensor = value.getTensor<uint64_t>(); tensor) {
              return tensor.value().values;
            } else {
              throw std::invalid_argument(
                  "Value isn't a tensor or has an unsupported "
                  "bitwidth");
            }
          },
          "Return the data from a Value, assuming it is a tensor of unsigned "
          "elements.")
      .def(
          "get_signed_tensor_data",
          [](Value &value) {
            if (auto tensor = value.getTensor<int8_t>(); tensor) {
              Tensor<int64_t> out = (Tensor<int64_t>)tensor.value();
              return out.values;
            } else if (auto tensor = value.getTensor<int16_t>(); tensor) {
              Tensor<int64_t> out = (Tensor<int64_t>)tensor.value();
              return out.values;
            } else if (auto tensor = value.getTensor<int32_t>(); tensor) {
              Tensor<int64_t> out = (Tensor<int64_t>)tensor.value();
              return out.values;
            } else if (auto tensor = value.getTensor<int64_t>(); tensor) {
              return tensor.value().values;
            } else {
              throw std::invalid_argument(
                  "Value isn't a tensor or has an unsupported "
                  "bitwidth");
            }
          },
          "Return the data from a Value, assuming it is a tensor of signed "
          "elements.")
      .def(
          "get_shape",
          [](Value &value) {
            std::vector<size_t> dims = value.getDimensions();
            return std::vector<int64_t>{dims.begin(), dims.end()};
          },
          "Return the shape of a Value.")
      .def(
          "is_scalar", [](Value &value) { return value.isScalar(); },
          "Return if the value is a scalar (as opposed to a tensor).")
      .def(
          "is_signed", [](Value &value) { return value.isSigned(); },
          "Return if the value has signed elements.")
      .def(
          "get_unsigned_scalar",
          [](Value &value) {
            if (value.isScalar() && value.hasElementType<uint64_t>()) {
              return value.getTensor<uint64_t>()->values[0];
            }
            throw std::invalid_argument("Value isn't an u64 scalar");
          },
          "Return the scalar from a Value, assuming it is an u64 scalar.")
      .def(
          "get_signed_scalar",
          [](Value &value) {
            if (value.isScalar() && value.hasElementType<int64_t>()) {
              return value.getTensor<int64_t>()->values[0];
            }
            throw std::invalid_argument("Value isn't a i64 scalar");
          },
          "Return the scalar from a Value, assuming it is an i64 scalar.")
      .doc() = "Private / Runtime value.";

  // ------------------------------------------------------------------------------//
  // PUBLIC ARGUMENTS //
  // ------------------------------------------------------------------------------//

  struct PublicArguments {
    std::vector<TransportValue> values;
  };
  pybind11::class_<PublicArguments, std::unique_ptr<PublicArguments>>(
      m, "PublicArguments")
      .def(init([](std::vector<TransportValue> &buffers) {
             return PublicArguments{buffers};
           }),
           arg("transport_values"))
      .def_static(
          "deserialize",
          [](const pybind11::bytes &buffer) {
            auto publicArgumentsProto =
                Message<concreteprotocol::PublicArguments>();
            if (publicArgumentsProto.readBinaryFromString(buffer)
                    .has_failure()) {
              throw std::runtime_error(
                  "Failed to deserialize public arguments.");
            }
            std::vector<TransportValue> values;
            for (auto arg : publicArgumentsProto.asReader().getArgs()) {
              values.push_back(arg);
            }
            PublicArguments output{values};
            return std::make_unique<PublicArguments>(std::move(output));
          },
          "Deserializes a PublicArguments from bytes.", arg("byte"))
      .def(
          "serialize",
          [](PublicArguments &publicArgument) {
            auto publicArgumentsSerialize =
                [](PublicArguments &publicArguments) {
                  auto publicArgumentsProto =
                      Message<concreteprotocol::PublicArguments>();
                  auto argBuilder = publicArgumentsProto.asBuilder().initArgs(
                      publicArguments.values.size());
                  for (size_t i = 0; i < publicArguments.values.size(); i++) {
                    argBuilder.setWithCaveats(
                        i, publicArguments.values[i].asReader());
                  }
                  auto maybeBuffer = publicArgumentsProto.writeBinaryToString();
                  if (maybeBuffer.has_failure()) {
                    throw std::runtime_error(
                        "Failed to serialize public arguments.");
                  }
                  return maybeBuffer.value();
                };
            return pybind11::bytes(publicArgumentsSerialize(publicArgument));
          },
          "Serialize a PublicArguments to bytes.")
      .doc() = "Public arguments to be sent from the client to the server "
               "before execution.";

  // ------------------------------------------------------------------------------//
  // PUBLIC RESULT //
  // ------------------------------------------------------------------------------//
  struct PublicResults {
    std::vector<TransportValue> values;
  };
  pybind11::class_<PublicResults>(m, "PublicResults")
      .def(init([]() -> PublicResults {
        throw std::runtime_error("Explicit construction forbidden.");
      }))
      .def_static(
          "deserialize",
          [](const pybind11::bytes &buffer) {
            auto publicResultsProto =
                Message<concreteprotocol::PublicResults>();
            if (publicResultsProto.readBinaryFromString(buffer).has_failure()) {
              throw std::runtime_error("Failed to deserialize public results.");
            }
            std::vector<TransportValue> values;
            for (auto res : publicResultsProto.asReader().getResults()) {
              values.push_back(res);
            }
            PublicResults output{values};
            return std::make_unique<PublicResults>(std::move(output));
          },
          "Deserialize a PublicResults from bytes", arg("bytes"))
      .def(
          "serialize",
          [](PublicResults &publicResult) {
            auto publicResultSerialize = [](PublicResults &publicResult) {
              std::string buffer;
              auto publicResultsProto =
                  Message<concreteprotocol::PublicResults>();
              auto resBuilder = publicResultsProto.asBuilder().initResults(
                  publicResult.values.size());
              for (size_t i = 0; i < publicResult.values.size(); i++) {
                resBuilder.setWithCaveats(i, publicResult.values[i].asReader());
              }
              auto maybeBuffer = publicResultsProto.writeBinaryToString();
              if (maybeBuffer.has_failure()) {
                throw std::runtime_error("Failed to serialize public results.");
              }
              return maybeBuffer.value();
            };
            return pybind11::bytes(publicResultSerialize(publicResult));
          },
          "Serialize a PublicResults to bytes.")
      .def(
          "n_values",
          [](const PublicResults &publicResult) {
            return publicResult.values.size();
          },
          "Return the number of values.")
      .def(
          "get_value",
          [](PublicResults &publicResult, size_t position) {
            if (position >= publicResult.values.size()) {
              throw std::runtime_error("Failed to get public result value.");
            }
            return publicResult.values[position];
          },
          "Get the `n`-th value from the results", arg("n"))
      .doc() = "Public results to be sent from the server to the client after "
               "execution.";

  // ------------------------------------------------------------------------------//
  // SERVER CIRCUIT //
  // ------------------------------------------------------------------------------//

  pybind11::class_<ServerCircuit>(m, "ServerCircuit")
      .def(init([]() -> ServerCircuit {
        throw std::runtime_error("Explicit construction forbidden.");
      }))
      .def(
          "call",
          [](ServerCircuit &circuit, std::vector<TransportValue> args,
             ServerKeyset keyset) {
            SignalGuard signalGuard;
            pybind11::gil_scoped_release release;
            GET_OR_THROW_RESULT(auto output, circuit.call(keyset, args));
            return output;
          },
          "Perform circuit call with `args` arguments using the `keyset` "
          "ServerKeyset.",
          arg("args"), arg("keyset"))
      .def(
          "simulate",
          [](ServerCircuit &circuit, std::vector<TransportValue> &args) {
            pybind11::gil_scoped_release release;
            GET_OR_THROW_RESULT(auto output, circuit.simulate(args));
            return output;
          },
          "Perform circuit simulation with `args` arguments.", arg("args"))
      .doc() = "Server-side / Evaluation circuit.";

  // ------------------------------------------------------------------------------//
  // SERVER PROGRAM //
  // ------------------------------------------------------------------------------//

  pybind11::class_<ServerProgram>(m, "ServerProgram")
      .def(init([](Library &library, bool useSimulation) {
             auto sharedLibPath = library.getSharedLibraryPath();
             GET_OR_THROW_RESULT(
                 auto result,
                 ServerProgram::load(library.getProgramInfo().asReader(),
                                     sharedLibPath, useSimulation));
             return result;
           }),
           arg("library"), arg("use_simulation"))
      .def(
          "get_server_circuit",
          [](ServerProgram &program, const std::string &circuitName) {
            GET_OR_THROW_RESULT(auto result,
                                program.getServerCircuit(circuitName));
            return result;
          },
          "Return the `circuit` ServerCircuit.", arg("circuit"))
      .doc() = "Server-side / Evaluation program.";

  // ------------------------------------------------------------------------------//
  // CLIENT CIRCUIT //
  // ------------------------------------------------------------------------------//

  pybind11::class_<ClientCircuit>(m, "ClientCircuit")
      .def(
          "prepare_input",
          [](ClientCircuit &circuit, Value arg, size_t pos) {
            if (pos > circuit.getCircuitInfo().asReader().getInputs().size()) {
              throw std::runtime_error("Unknown position.");
            }
            auto info = circuit.getCircuitInfo().asReader().getInputs()[pos];
            auto typeTransformer = getPythonTypeTransformer(info);
            GET_OR_THROW_RESULT(
                auto ok, circuit.prepareInput(typeTransformer(arg), pos));
            return ok;
          },
          "Prepare a `pos` positional arguments `arg` to be sent to server. ",
          arg("arg"), arg("pos"))
      .def(
          "process_output",
          [](ClientCircuit &circuit, TransportValue result, size_t pos) {
            GET_OR_THROW_RESULT(auto ok, circuit.processOutput(result, pos));
            return ok;
          },
          "Process a `pos` positional result `result` retrieved from server. ",
          arg("result"), arg("pos"))
      .def(
          "simulate_prepare_input",
          [](ClientCircuit &circuit, Value arg, size_t pos) {
            if (pos > circuit.getCircuitInfo().asReader().getInputs().size()) {
              throw std::runtime_error("Unknown position.");
            }
            auto info = circuit.getCircuitInfo().asReader().getInputs()[pos];
            auto typeTransformer = getPythonTypeTransformer(info);
            GET_OR_THROW_RESULT(auto ok,
                                circuit.simulatePrepareInput(arg, pos));
            return ok;
          },
          "SIMULATE preparation of `pos` positional argument `arg` to be sent "
          "to server. DOES NOT ENCRYPT.",
          arg("arg"), arg("pos"))
      .def(
          "simulate_process_output",
          [](ClientCircuit &circuit, TransportValue result, size_t pos) {
            GET_OR_THROW_RESULT(auto ok,
                                circuit.simulateProcessOutput(result, pos));
            return ok;
          },
          "SIMULATE processing of `pos` positional result `result` retrieved "
          "from server.",
          arg("result"), arg("pos"))
      .doc() = "Client-side / Encryption circuit.";

  // ------------------------------------------------------------------------------//
  // CLIENT PROGRAM //
  // ------------------------------------------------------------------------------//

  pybind11::class_<ClientProgram>(m, "ClientProgram")
      .def_static(
          "create_encrypted",
          [](ProgramInfo programInfo, Keyset keyset) {
            GET_OR_THROW_RESULT(
                auto clientProgram,
                ClientProgram::create_encrypted(
                    programInfo.programInfo, keyset.client,
                    std::make_shared<EncryptionCSPRNG>(EncryptionCSPRNG(0))));
            return clientProgram;
          },
          "Create an encrypted (as opposed to simulated) ClientProgram.",
          arg("program_info"), arg("keyset"))
      .def_static(
          "create_simulated",
          [](ProgramInfo &programInfo) {
            GET_OR_THROW_RESULT(
                auto clientProgram,
                ClientProgram::create_simulated(
                    programInfo.programInfo,
                    std::make_shared<EncryptionCSPRNG>(EncryptionCSPRNG(0))));
            return clientProgram;
          },
          "Create a simulated (as opposed to encrypted) ClientProgram. DOES "
          "NOT PERFORM ENCRYPTION OF VALUES.",
          arg("program_info"))
      .def(
          "get_client_circuit",
          [](ClientProgram &program,
             const std::string &circuitName) -> ClientCircuit {
            GET_OR_THROW_RESULT(auto result,
                                program.getClientCircuit(circuitName));
            return result;
          },
          "Return the `circuit` ClientCircuit.", arg("circuit"))
      .doc() = "Client-side / Encryption program";

  m.def("import_tfhers_fheuint8",
        [](const pybind11::bytes &serialized_fheuint,
           TfhersFheIntDescription info, uint32_t encryptionKeyId,
           double encryptionVariance) {
          const std::string &buffer_str = serialized_fheuint;
          std::vector<uint8_t> buffer(buffer_str.begin(), buffer_str.end());
          auto arrayRef = llvm::ArrayRef<uint8_t>(buffer);
          auto valueOrError = ::concretelang::clientlib::importTfhersFheUint8(
              arrayRef, info, encryptionKeyId, encryptionVariance);
          if (valueOrError.has_error()) {
            throw std::runtime_error(valueOrError.error().mesg);
          }
          return TransportValue{valueOrError.value()};
        });

  m.def("export_tfhers_fheuint8",
        [](TransportValue fheuint, TfhersFheIntDescription info) {
          auto result =
              ::concretelang::clientlib::exportTfhersFheUint8(fheuint, info);
          if (result.has_error()) {
            throw std::runtime_error(result.error().mesg);
          }
          return result.value();
        });

  m.def("get_tfhers_fheuint8_description",
        [](const pybind11::bytes &serialized_fheuint) {
          const std::string &buffer_str = serialized_fheuint;
          std::vector<uint8_t> buffer(buffer_str.begin(), buffer_str.end());
          auto arrayRef = llvm::ArrayRef<uint8_t>(buffer);
          auto info =
              ::concretelang::clientlib::getTfhersFheUint8Description(arrayRef);
          if (info.has_error()) {
            throw std::runtime_error(info.error().mesg);
          }
          return info.value();
        });
}
