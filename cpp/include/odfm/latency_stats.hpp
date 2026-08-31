// latency_stats.hpp — latency accounting for a real-time perception runtime.
//
// Two things this fixes about how latency is currently reported in this repo:
//
// 1. The README quotes p50 and p95 only. For an AV stack the tail is the number
//    that matters: a p99.9 spike is a dropped frame at 30 Hz, and a frame lost
//    during a cut-in is the case the system exists to handle. This records
//    p99 and p99.9 and an explicit jitter ratio.
//
// 2. Percentiles here use numpy's default 'linear' interpolation, deliberately,
//    so a C++ number and a Python number computed over the same samples are
//    directly comparable. A C++ harness that reports "p95" under a different
//    percentile convention than the Python one silently invents a discrepancy.
//
// C++17, header-only, no dependencies.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace odfm {

class LatencyStats {
 public:
  explicit LatencyStats(std::size_t reserve = 1024) { samples_.reserve(reserve); }

  void add(double ms) {
    samples_.push_back(ms);
    sorted_ = false;  // invalidate the lazy sort cache, or every later
                      // percentile silently reports a stale distribution
  }
  std::size_t count() const noexcept { return samples_.size(); }
  bool empty() const noexcept { return samples_.empty(); }

  /// Percentile with linear interpolation between order statistics, matching
  /// numpy.percentile(..., method="linear"). q is in [0, 100].
  double percentile(double q) const {
    if (samples_.empty()) return std::nan("");
    if (!sorted_) {
      sorted_cache_ = samples_;
      std::sort(sorted_cache_.begin(), sorted_cache_.end());
      sorted_ = true;
    }
    const std::size_t n = sorted_cache_.size();
    if (n == 1) return sorted_cache_[0];
    const double pos = (q / 100.0) * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
    const double frac = pos - static_cast<double>(lo);
    return sorted_cache_[lo] + frac * (sorted_cache_[hi] - sorted_cache_[lo]);
  }

  double min() const { return percentile(0.0); }
  double max() const { return percentile(100.0); }
  double p50() const { return percentile(50.0); }
  double p95() const { return percentile(95.0); }
  double p99() const { return percentile(99.0); }
  double p999() const { return percentile(99.9); }

  double mean() const {
    if (samples_.empty()) return std::nan("");
    double s = 0.0;
    for (double v : samples_) s += v;
    return s / static_cast<double>(samples_.size());
  }

  /// Sample standard deviation (n-1 denominator).
  double stddev() const {
    const std::size_t n = samples_.size();
    if (n < 2) return std::nan("");
    const double m = mean();
    double acc = 0.0;
    for (double v : samples_) acc += (v - m) * (v - m);
    return std::sqrt(acc / static_cast<double>(n - 1));
  }

  /// p99/p50. A real-time system wants this close to 1.0; a large ratio means
  /// the mean is hiding stalls.
  double jitter_ratio() const {
    const double m = p50();
    return (m > 0.0) ? p99() / m : std::nan("");
  }

  double throughput_fps() const {
    const double m = mean();
    return (m > 0.0) ? 1000.0 / m : std::nan("");
  }

  /// Fraction of samples exceeding a per-frame budget. This is the SLO check:
  /// at 30 Hz the budget is 33.3 ms.
  double deadline_miss_rate(double budget_ms) const {
    if (samples_.empty()) return std::nan("");
    std::size_t over = 0;
    for (double v : samples_) if (v > budget_ms) ++over;
    return static_cast<double>(over) / static_cast<double>(samples_.size());
  }

  std::string to_json(double budget_ms = 33.3) const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(6) << "{"
      << "\"count\":" << count()
      << ",\"min_ms\":" << min()
      << ",\"p50_ms\":" << p50()
      << ",\"p95_ms\":" << p95()
      << ",\"p99_ms\":" << p99()
      << ",\"p999_ms\":" << p999()
      << ",\"max_ms\":" << max()
      << ",\"mean_ms\":" << mean()
      << ",\"stddev_ms\":" << stddev()
      << ",\"jitter_p99_over_p50\":" << jitter_ratio()
      << ",\"throughput_fps\":" << throughput_fps()
      << ",\"deadline_budget_ms\":" << budget_ms
      << ",\"deadline_miss_rate\":" << deadline_miss_rate(budget_ms)
      << "}";
    return o.str();
  }

 private:
  std::vector<double> samples_;
  mutable std::vector<double> sorted_cache_;
  mutable bool sorted_{false};
};

}  // namespace odfm
