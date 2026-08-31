// odfm_runner.cpp — a two-thread real-time perception node.
//
// WHY THIS IS DIFFERENT FROM A LATENCY BENCHMARK
// ----------------------------------------------
// `odfm_parity_check` measures how long model.forward() takes. That is *model*
// latency, and it is not the number a planner experiences. What matters on a
// vehicle is END-TO-END staleness: how old is the frame that produced the
// occupancy grid the planner is acting on right now?
//
// Those differ the moment the sensor produces frames faster than inference can
// consume them. Model latency stays flat; end-to-end latency grows without
// bound as the queue backs up. A system reporting "14 ms inference" can be
// handing the planner 400 ms old perception.
//
// This binary runs a producer thread at a fixed sensor rate and a consumer
// thread doing inference, and measures three separate quantities:
//
//   queue wait   dequeue time  - capture time
//   inference    forward()
//   end-to-end   result time   - capture time      <- the one that matters
//
// TWO BACKPRESSURE POLICIES
// -------------------------
//   --policy queue    FIFO SpscRing. Every frame is processed. Under overload
//                     the queue fills, staleness grows, and the producer must
//                     drop at the source.
//   --policy latest   LatestFrame seqlock. The consumer always takes the newest
//                     frame and skips whatever arrived while it was busy.
//                     Staleness stays bounded; coverage is sacrificed.
//
// Neither is correct in general. Perception wants `latest` (a 400 ms old
// obstacle is worse than no obstacle); a recording or an event detector wants
// `queue`. Being able to measure the difference is the point.
//
// Usage:
//   ./odfm_runner --model outputs/artifacts/opendrivefm_v11.pt \
//                 --reference outputs/artifacts/parity_reference.pt \
//                 --fps 30 --seconds 10 --policy latest --threads 4
#include <torch/script.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "odfm/latency_stats.hpp"
#include "odfm/spsc_ring.hpp"

namespace {

using clk = std::chrono::steady_clock;

std::string arg_str(int argc, char** argv, const std::string& k, const std::string& d) {
  for (int i = 1; i + 1 < argc; ++i) if (k == argv[i]) return argv[i + 1];
  return d;
}
double arg_num(int argc, char** argv, const std::string& k, double d) {
  for (int i = 1; i + 1 < argc; ++i) if (k == argv[i]) return std::atof(argv[i + 1]);
  return d;
}

/// POD frame metadata. A real pipeline passes an index into a preallocated
/// image pool, never the pixels: copying a 6-camera frame through a queue would
/// dominate every latency number in this file.
struct FrameRef {
  std::uint64_t seq;
  std::int64_t capture_ns;
};

double ms_since(std::int64_t start_ns) {
  const auto now = clk::now().time_since_epoch().count();
  return static_cast<double>(now - start_ns) / 1e6;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_path =
      arg_str(argc, argv, "--model", "outputs/artifacts/opendrivefm_v11.pt");
  const std::string ref_path =
      arg_str(argc, argv, "--reference", "outputs/artifacts/parity_reference.pt");
  const double fps = arg_num(argc, argv, "--fps", 30.0);
  const double seconds = arg_num(argc, argv, "--seconds", 10.0);
  const std::string policy = arg_str(argc, argv, "--policy", "latest");
  const int threads = static_cast<int>(arg_num(argc, argv, "--threads", 4));
  const std::size_t depth =
      static_cast<std::size_t>(arg_num(argc, argv, "--queue-depth", 8));

  torch::set_num_threads(threads);
  torch::NoGradGuard no_grad;

  torch::jit::script::Module model, refs;
  try {
    model = torch::jit::load(model_path);
    refs = torch::jit::load(ref_path);
  } catch (const c10::Error& e) {
    std::cerr << "Failed to load: " << e.what() << "\n";
    return 2;
  }
  model.eval();
  const auto x = refs.attr("input_x").toTensor();
  const auto vel = refs.attr("input_velocity").toTensor();
  std::vector<torch::jit::IValue> inputs{x, vel};

  const double period_ms = 1000.0 / fps;
  const auto total_frames = static_cast<std::uint64_t>(fps * seconds);

  std::cout << "odfm_runner\n"
            << "  policy       : " << policy << "\n"
            << "  sensor rate  : " << fps << " Hz (period " << period_ms << " ms)\n"
            << "  duration     : " << seconds << " s (" << total_frames << " frames)\n"
            << "  torch threads: " << threads << "\n"
            << "  queue depth  : " << depth << "\n\n";

  for (int i = 0; i < 10; ++i) (void)model.forward(inputs);   // warm up

  odfm::SpscRing<FrameRef> ring(depth);
  odfm::LatestFrame<FrameRef> latest;
  std::atomic<bool> producing{true};
  std::atomic<std::uint64_t> produced{0}, dropped_full{0};

  // Waiting strategy. Two earlier versions were both wrong, and the pipeline's
  // own numbers showed it:
  //   yield() spin  -> the idle consumer stayed runnable and competed with the
  //                    inference threads. At 30 Hz this raised inference p50
  //                    from 13.93 to 15.20 ms and quadrupled its sd.
  //   sleep_for(200us) -> macOS rounds short sleeps up to ~1 ms and the thread
  //                    woke thousands of times a second; p50 recovered to
  //                    14.32 ms but max blew out from 21.9 to 49.6 ms.
  // A condition variable lets the consumer block until the producer actually
  // has something, consuming no CPU in between.
  std::mutex wait_mu;
  std::condition_variable wait_cv;
  std::atomic<std::uint64_t> produced_seq{0};

  std::thread producer([&] {
    const auto t0 = clk::now();
    for (std::uint64_t i = 0; i < total_frames; ++i) {
      std::this_thread::sleep_until(
          t0 + std::chrono::microseconds(
                   static_cast<long long>(i * period_ms * 1000.0)));
      FrameRef f{i, clk::now().time_since_epoch().count()};
      if (policy == "latest") {
        latest.store(f);                       // writer never blocks or fails
      } else if (!ring.push(f)) {
        dropped_full.fetch_add(1, std::memory_order_relaxed);
      }
      produced.fetch_add(1, std::memory_order_relaxed);
      // Publish then notify. The producer never takes the mutex, so it is still
      // wait-free; the consumer's bounded wait below covers the narrow window
      // where a notify lands just before the consumer starts waiting.
      produced_seq.fetch_add(1, std::memory_order_release);
      wait_cv.notify_one();
    }
    producing.store(false, std::memory_order_release);
    wait_cv.notify_all();
  });

  odfm::LatencyStats queue_wait, infer, end_to_end;
  std::uint64_t processed = 0, skipped_stale = 0;
  std::uint64_t last_seq = UINT64_MAX;
  std::uint64_t idle_spins = 0;

  std::thread consumer([&] {
    FrameRef f{};
    while (true) {
      bool have = false;
      if (policy == "latest") {
        if (latest.load(f) && f.seq != last_seq && f.capture_ns != 0) {
          // Count every frame produced since the last one we handled: those
          // were skipped, which is the cost of the freshness policy.
          if (last_seq != UINT64_MAX && f.seq > last_seq + 1)
            skipped_stale += (f.seq - last_seq - 1);
          last_seq = f.seq;
          have = true;
        }
      } else {
        have = ring.pop(f);
      }

      if (!have) {
        if (!producing.load(std::memory_order_acquire) &&
            (policy == "latest" || ring.size_approx() == 0)) break;
        ++idle_spins;
        const std::uint64_t seen = produced_seq.load(std::memory_order_acquire);
        std::unique_lock<std::mutex> lk(wait_mu);
        // Bounded wait: the predicate is re-checked on every wakeup, and the
        // 2 ms cap guarantees forward progress even if a notify is missed
        // because the producer does not hold this mutex.
        wait_cv.wait_for(lk, std::chrono::milliseconds(2), [&] {
          return !producing.load(std::memory_order_acquire) ||
                 produced_seq.load(std::memory_order_acquire) != seen;
        });
        continue;
      }

      queue_wait.add(ms_since(f.capture_ns));
      const auto t0 = clk::now();
      (void)model.forward(inputs);
      const auto t1 = clk::now();
      infer.add(std::chrono::duration<double, std::milli>(t1 - t0).count());
      end_to_end.add(ms_since(f.capture_ns));
      ++processed;
    }
  });

  producer.join();
  consumer.join();

  auto row = [](const char* name, const odfm::LatencyStats& s) {
    std::cout << std::left << std::setw(14) << name << std::right << std::fixed
              << std::setprecision(2) << std::setw(9) << s.p50()
              << std::setw(9) << s.p95() << std::setw(9) << s.p99()
              << std::setw(10) << s.max() << "\n";
  };

  std::cout << std::left << std::setw(14) << "stage (ms)" << std::right
            << std::setw(9) << "p50" << std::setw(9) << "p95" << std::setw(9)
            << "p99" << std::setw(10) << "max" << "\n"
            << std::string(51, '-') << "\n";
  row("queue wait", queue_wait);
  row("inference", infer);
  row("END-TO-END", end_to_end);

  const auto n_prod = produced.load();
  std::cout << "\nframes produced   : " << n_prod
            << "\nframes processed  : " << processed
            << " (" << std::setprecision(1)
            << (100.0 * processed / std::max<std::uint64_t>(n_prod, 1)) << "%)"
            << "\nskipped as stale  : " << skipped_stale
            << "\ndropped queue-full: " << dropped_full.load() + ring.dropped()
            << "\neffective rate    : " << std::setprecision(2)
            << (processed / seconds) << " Hz"
            << "\nconsumer idle waits: " << idle_spins
            << " (condition-variable blocks, not spins)\n";

  const double budget = period_ms;
  std::cout << "\nend-to-end staleness above one sensor period (" << budget
            << " ms): " << std::setprecision(1)
            << 100.0 * end_to_end.deadline_miss_rate(budget) << "%\n";

  if (end_to_end.p99() > 3.0 * infer.p99()) {
    std::cout << "\nWARNING: end-to-end p99 is more than 3x inference p99. The pipeline\n"
                 "is queue-bound, not compute-bound: the planner is acting on stale\n"
                 "perception even though inference looks fast. Reduce the sensor rate,\n"
                 "shrink the queue, or switch to --policy latest.\n";
  }
  std::cout << "\nJSON {\"policy\":\"" << policy << "\",\"fps\":" << fps
            << ",\"processed\":" << processed
            << ",\"skipped_stale\":" << skipped_stale
            << ",\"dropped_full\":" << (dropped_full.load() + ring.dropped())
            << ",\"inference\":" << infer.to_json(budget)
            << ",\"end_to_end\":" << end_to_end.to_json(budget) << "}\n";
  return 0;
}
