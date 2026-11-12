#include "Taskflow/ProgramRuntimeCompatibility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace {
const char *source = R"mlir(
module {
  "func.func"() <{sym_name = "compat", function_type = (i32) -> i16}> ({
  ^bb0(%x: i32):
    %sum = "arith.addi"(%x, %x) <{overflowFlags = #arith.overflow<nsw>}> : (i32, i32) -> i32
    %first = "arith.trunci"(%sum) <{overflowFlags = #arith.overflow<none>}> {test.marker = "retain"} : (i32) -> i16
    %wide = "arith.extui"(%first) : (i16) -> i32
    %second = "arith.trunci"(%wide) <{overflowFlags = #arith.overflow<none>}> : (i32) -> i16
    "func.return"(%second) : (i16) -> ()
  }) : () -> ()
}
)mlir";
std::string changed(std::string input, llvm::StringRef from,
                    llvm::StringRef to) {
  auto position = input.find(from.str());
  if (position == std::string::npos)
    llvm::report_fatal_error("missing negative-test mutation site");
  input.replace(position, from.size(), to.str());
  return input;
}
} // namespace

int main() {
  // Independent standard SHA-256 vector verifies the receipt byte hasher.
  if (orbit::taskflow::programRuntimeTextSHA256("abc") !=
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    return 6;
  std::string output, error;
  size_t removed = 0;
  if (!orbit::taskflow::normalizeProgramRuntimeDefaults(source, output, removed,
                                                        error) ||
      removed != 2)
    return 1;
  // A real registered parser must accept the normalized operations. Preserve
  // the unrelated nsw promise and ordinary attribute instead of stripping all
  // arithmetic flags or text that merely contains the name overflowFlags.
  mlir::MLIRContext registered;
  registered.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(output, &registered);
  if (!parsed || mlir::failed(mlir::verify(*parsed)) ||
      output.find("#arith.overflow<nsw>") == std::string::npos ||
      output.find("test.marker = \"retain\"") == std::string::npos)
    return 2;
  std::string second;
  if (!orbit::taskflow::normalizeProgramRuntimeDefaults(output, second, removed,
                                                        error) ||
      removed != 0 || second != output)
    return 3;
  if (orbit::taskflow::programRuntimeTextSHA256(source) ==
      orbit::taskflow::programRuntimeTextSHA256(output))
    return 7;
  const auto property = "<{overflowFlags = #arith.overflow<none>}>";
  for (const auto *replacement :
       {"<{overflowFlags = #arith.overflow<nsw>}>",
        "<{overflowFlags = #arith.overflow<nuw>}>",
        "<{overflowFlags = #arith.overflow<none>, unknown = 1 : i32}>",
        "<{overflowFlags = 0 : i32}>",
        "{overflowFlags = #arith.overflow<nsw>}"}) {
    auto negative = changed(source, property, replacement);
    if (replacement[0] == '{')
      negative = changed(negative, " {test.marker = \"retain\"}", "");
    output = "must be cleared";
    removed = 99;
    if (orbit::taskflow::normalizeProgramRuntimeDefaults(negative, output,
                                                         removed, error) ||
        !output.empty() || removed != 0 || error.empty())
      return 4;
  }
  // A missing default property is already compatible and must retain bytes.
  auto noDefault = changed(source, property, "");
  noDefault = changed(noDefault, property, "");
  if (!orbit::taskflow::normalizeProgramRuntimeDefaults(noDefault, output,
                                                        removed, error) ||
      removed || output != noDefault)
    return 5;
  llvm::outs() << "runtime compatibility: two exact defaults, preserved flags, "
                  "idempotence and five negative guards PASS\n";
}
