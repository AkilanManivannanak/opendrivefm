// odfm_parity_check.cpp — does the C++ deployment graph still compute what the
// validated Python model computed?
//
// WHY THIS EXISTS
// ---------------
// A perception model is trained and validated in Python and shipped in C++.
// When the two diverge -- a traced graph that froze a different branch, a
// LibTorch version bump, a changed thread count, a fused kernel with different
// rounding -- nothing crashes. The C++ binary keeps emitting plausible occupancy
// grids that no longer match the model anyone signed off on. Silent numerical
// drift between the validated model and the shipped one is one of the standard
// ways an AV programme ships a regression.
//
// This binary loads a reference bundle produced by scripts/export_torchscript.py
// (inputs plus the eager-mode outputs), runs the TorchScript graph on the same
// inputs in C++, and fails with a non-zero exit code if any output moves beyond
// tolerance. Drop it in CI and drift becomes a red build instead of a field
// incident.
//
// It also checks two things that are cheap here and expensive later:
//   * determinism -- the same input twice must give bit-identical output, or
//     no downstream metric is reproducible;
//   * tail latency -- p99 and p99.9 against a stated per-frame budget, because
//     a 30 Hz stack is defined by its worst frame, not its average one.
//
// Usage:
//   ./odfm_parity_check --model outputs/artifacts/opendrivefm_v11.pt \
//                       --reference outputs/artifacts/parity_reference.pt \
//                       --atol 1e-5 --rtol 1e-4 --iters 200 --budget-ms 33.3
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "odfm/latency_stats.hpp"

namespace {

std::string arg_str(int argc, char** argv, const std::string& key,
                    const std::string& fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (key == argv[i]) return argv[i + 1];
  return fallback;
}

double arg_num(int argc, char** argv, const std::string& key, double fallback) {
  for (int i = 1; i + 1 < argc; ++i)
    if (key == argv[i]) return std::atof(argv[i + 1]);
  return fallback;
}

struct Comparison {
  std::string name;
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;   // over significant elements only, see below
  double ref_absmax = 0.0;
  int64_t n_significant = 0;
  bool passed = false;
};

/// Compare a C++ output against the Python reference for one tensor.
Comparison compare(const std::string& name, const torch::Tensor& got,
                   const torch::Tensor& ref, double atol, double rtol,
                   double rel_floor) {
  Comparison c;
  c.name = name;

  if (got.sizes() != ref.sizes()) {
    std::cerr << "  SHAPE MISMATCH for " << name << ": C++ " << got.sizes()
              << " vs Python " << ref.sizes() << "\n";
    return c;  // passed stays false
  }

  const auto a = got.to(torch::kDouble).contiguous();
  const auto b = ref.to(torch::kDouble).contiguous();
  const auto diff = (a - b).abs();

  c.max_abs_diff = diff.max().item<double>();
  c.ref_absmax = b.abs().max().item<double>();

  // Relative error is only meaningful where the reference is meaningfully
  // non-zero. Occupancy logits pass through zero, so dividing by |ref| there
  // produces enormous ratios from absolute differences of ~1e-6 and hides the
  // real signal. Restrict the relative statistic to elements above a floor and
  // report how many elements that was, so the number can be interpreted.
  const auto significant = b.abs() > rel_floor;
  c.n_significant = significant.sum().item<int64_t>();
  if (c.n_significant > 0) {
    const auto rel = diff.masked_select(significant) /
                     b.abs().masked_select(significant);
    c.max_rel_diff = rel.max().item<double>();
  }

  c.passed = torch::allclose(a, b, rtol, atol);
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_path =
      arg_str(argc, argv, "--model", "outputs/artifacts/opendrivefm_v11.pt");
  const std::string ref_path =
      arg_str(argc, argv, "--reference", "outputs/artifacts/parity_reference.pt");
  const double atol = arg_num(argc, argv, "--atol", 1e-5);
  const double rtol = arg_num(argc, argv, "--rtol", 1e-4);
  const int iters = static_cast<int>(arg_num(argc, argv, "--iters", 200));
  const int warmup = static_cast<int>(arg_num(argc, argv, "--warmup", 20));
  const double budget_ms = arg_num(argc, argv, "--budget-ms", 33.3);
  const int threads = static_cast<int>(arg_num(argc, argv, "--threads", 1));
  const double rel_floor = arg_num(argc, argv, "--rel-floor", 1e-3);

  // Pinned for reproducibility: latency and, on some kernels, reduction order
  // both depend on the thread count. A parity number measured under a different
  // thread count is not comparable.
  torch::set_num_threads(threads);
  torch::NoGradGuard no_grad;

  torch::jit::script::Module model, refs;
  try {
    model = torch::jit::load(model_path);
    refs = torch::jit::load(ref_path);
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load:\n" << e.what() << "\n"
              << "Run scripts/export_torchscript.py first.\n";
    return 2;
  }
  model.eval();

  const auto x = refs.attr("input_x").toTensor();
  const auto vel = refs.attr("input_velocity").toTensor();
  const std::vector<std::pair<std::string, torch::Tensor>> reference = {
      {"occupancy", refs.attr("ref_occupancy").toTensor()},
      {"trajectory", refs.attr("ref_trajectory").toTensor()},
      {"trust", refs.attr("ref_trust").toTensor()},
  };

  std::cout << "odfm_parity_check\n"
            << "  model      : " << model_path << "\n"
            << "  reference  : " << ref_path << "\n"
            << "  input x    : " << x.sizes() << "\n"
            << "  threads    : " << threads << "\n"
            << "  tolerance  : atol=" << atol << " rtol=" << rtol << "\n\n";

  std::vector<torch::jit::IValue> inputs{x, vel};

  torch::jit::IValue raw;
  try {
    raw = model.forward(inputs);
  } catch (const c10::Error& e) {
    std::cerr << "Forward pass failed:\n" << e.what() << "\n";
    return 2;
  }
  if (!raw.isTuple()) {
    std::cerr << "Expected the graph to return a tuple of 3 tensors. Re-export "
                 "with the current scripts/export_torchscript.py.\n";
    return 2;
  }
  const auto tuple1 = raw.toTuple();
  const auto& out = tuple1->elements();
  if (out.size() != reference.size()) {
    std::cerr << "Expected " << reference.size() << " outputs, got " << out.size()
              << ".\n";
    return 2;
  }

  // ── 1. Numerical parity against the Python reference ───────────────────────
  bool all_passed = true;
  std::vector<Comparison> comps;
  std::cout << std::left << std::setw(14) << "output" << std::right
            << std::setw(16) << "max|C++ - py|" << std::setw(16) << "max rel*"
            << std::setw(14) << "|ref|max" << std::setw(12) << "n signif" << std::setw(10) << "result" << "\n"
            << std::string(82, '-') << "\n";
  for (size_t i = 0; i < reference.size(); ++i) {
    const auto c = compare(reference[i].first, out[i].toTensor(),
                           reference[i].second, atol, rtol, rel_floor);
    comps.push_back(c);
    all_passed = all_passed && c.passed;
    std::cout << std::left << std::setw(14) << c.name << std::right
              << std::scientific << std::setprecision(3) << std::setw(16)
              << c.max_abs_diff << std::setw(16) << c.max_rel_diff
              << std::setw(14) << c.ref_absmax << std::setw(12) << c.n_significant
              << std::setw(10) << (c.passed ? "PASS" : "FAIL") << "\n";
  }

  // ── 2. Determinism ─────────────────────────────────────────────────────────
  // NOTE: the IValue returned by forward() must be held in a named variable.
  // Writing `model.forward(inputs).toTuple()->elements()` binds a reference into
  // a Tuple owned only by temporaries, which are destroyed at the end of the
  // full expression -- the reference dangles and the next read segfaults.
  const torch::jit::IValue raw2 = model.forward(inputs);
  const auto tuple2 = raw2.toTuple();
  const auto& second = tuple2->elements();
  bool deterministic = true;
  for (size_t i = 0; i < reference.size(); ++i)
    deterministic = deterministic &&
                    out[i].toTensor().equal(second[i].toTensor());
  std::cout << "\n  * max rel is computed only over elements with |ref| > "
            << rel_floor << "\n";

  std::cout << "\ndeterminism   : " << (deterministic
      ? "PASS (identical outputs on a repeated forward)"
      : "FAIL (repeated forward differs -- no downstream metric is reproducible)")
            << "\n";
  all_passed = all_passed && deterministic;

  // ── 3. Tail latency against a stated budget ────────────────────────────────
  for (int i = 0; i < warmup; ++i) (void)model.forward(inputs);
  odfm::LatencyStats lat(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    (void)model.forward(inputs);
    const auto t1 = std::chrono::steady_clock::now();
    lat.add(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  std::cout << std::fixed << std::setprecision(3)
            << "\nlatency over " << iters << " iterations (warmup " << warmup << ")\n"
            << "  p50   " << lat.p50() << " ms\n"
            << "  p95   " << lat.p95() << " ms\n"
            << "  p99   " << lat.p99() << " ms\n"
            << "  p99.9 " << lat.p999() << " ms\n"
            << "  max   " << lat.max() << " ms\n"
            << "  jitter p99/p50 " << lat.jitter_ratio() << "\n"
            << "  throughput     " << lat.throughput_fps() << " FPS\n"
            << "  deadline misses at " << budget_ms << " ms budget: "
            << 100.0 * lat.deadline_miss_rate(budget_ms) << "%\n";

  // Machine-readable line for CI to archive.
  std::cout << "\nJSON " << lat.to_json(budget_ms) << "\n";

  std::cout << "\n" << (all_passed
      ? "PARITY OK: the C++ graph matches the validated Python model.\n"
      : "PARITY FAILED: the shipped graph no longer matches the model that was\n"
        "validated. Do not deploy this artifact until the divergence is explained.\n");
  return all_passed ? 0 : 1;
}
