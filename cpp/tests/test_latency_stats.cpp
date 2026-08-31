// Unit tests for odfm::LatencyStats. Expected percentile values are computed by
// hand from numpy's 'linear' interpolation definition so that these tests prove
// C++/Python agreement rather than merely self-consistency.
#include "odfm/latency_stats.hpp"

#include <cmath>
#include <cstdio>
#include <string>

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
  if (!ok) { std::printf("  FAIL: %s\n", what.c_str()); ++g_failures; }
  else     { std::printf("  ok:   %s\n", what.c_str()); }
}

static void check_near(double got, double want, double tol, const std::string& what) {
  const bool ok = std::fabs(got - want) <= tol;
  if (!ok) std::printf("  FAIL: %s (got %.9f, want %.9f)\n", what.c_str(), got, want);
  else     std::printf("  ok:   %s\n", what.c_str());
  if (!ok) ++g_failures;
}

int main() {
  std::printf("LatencyStats\n");

  // numpy.percentile([1..10], 50) == 5.5 with linear interpolation:
  // pos = 0.5*(10-1) = 4.5 -> 5 + 0.5*(6-5) = 5.5
  {
    odfm::LatencyStats s;
    for (int i = 1; i <= 10; ++i) s.add(i);
    check_near(s.p50(), 5.5, 1e-12, "p50 of 1..10 matches numpy linear (5.5)");
    // pos = 0.95*9 = 8.55 -> 9 + 0.55*(10-9) = 9.55
    check_near(s.p95(), 9.55, 1e-12, "p95 of 1..10 matches numpy linear (9.55)");
    check_near(s.min(), 1.0, 1e-12, "min");
    check_near(s.max(), 10.0, 1e-12, "max");
    check_near(s.mean(), 5.5, 1e-12, "mean");
    // sample stddev of 1..10 = sqrt(110/12) ~= 3.02765
    check_near(s.stddev(), 3.0276503541, 1e-9, "sample stddev (n-1)");
  }

  // Regression: percentile() sorts lazily into a cache. Adding samples after a
  // percentile has been computed must invalidate that cache, or every later
  // number is silently stale.
  {
    odfm::LatencyStats s;
    for (int i = 1; i <= 10; ++i) s.add(i);
    (void)s.p50();
    for (int i = 11; i <= 20; ++i) s.add(i);
    check_near(s.p50(), 10.5, 1e-12, "p50 recomputed after add() (stale-cache guard)");
    check_near(s.max(), 20.0, 1e-12, "max recomputed after add()");
  }

  {
    odfm::LatencyStats s;
    for (int i = 0; i < 990; ++i) s.add(3.0);
    for (int i = 0; i < 10; ++i) s.add(80.0);   // 1% tail spike
    check(s.p50() < 4.0, "p50 unaffected by a 1% tail");
    check(s.p999() > 50.0, "p99.9 exposes the tail the mean hides");
    // NOTE: with exactly 1% of samples over budget, p99 lands ON the boundary
    // by construction (pos = 0.99*999 = 989.01), so p99/p50 is only ~1.26.
    // p99.9 is the statistic that actually exposes a 1% tail.
    check(s.jitter_ratio() > 1.2, "p99/p50 rises above 1.0 with a 1% tail");
    check(s.p999() / s.p50() > 5.0, "p99.9/p50 exposes the stall clearly");
    check_near(s.deadline_miss_rate(33.3), 0.01, 1e-12, "deadline miss rate at 30 Hz budget");
  }

  {
    odfm::LatencyStats s;
    check(std::isnan(s.p50()), "empty -> NaN, not 0");
    s.add(5.0);
    check_near(s.p95(), 5.0, 1e-12, "single sample");
    check(std::isnan(s.stddev()), "stddev of one sample is NaN, not 0");
  }

  {
    odfm::LatencyStats s;
    s.add(2.0); s.add(4.0);
    const std::string j = s.to_json();
    check(j.find("\"p999_ms\"") != std::string::npos, "json carries p99.9");
    check(j.find("\"deadline_miss_rate\"") != std::string::npos, "json carries SLO field");
  }

  std::printf(g_failures ? "\nLatencyStats: %d FAILURE(S)\n" : "\nLatencyStats: all passed\n",
              g_failures);
  return g_failures ? 1 : 0;
}
