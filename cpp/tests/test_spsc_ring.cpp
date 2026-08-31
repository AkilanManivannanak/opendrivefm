// Unit tests for odfm::SpscRing and odfm::LatestFrame.
//
// The concurrent tests are the point. A single-threaded test of a lock-free
// queue proves almost nothing: the failure modes are torn values, lost items
// and reordering under real contention on two cores.
#include "odfm/spsc_ring.hpp"

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

static int g_failures = 0;

static void check(bool ok, const std::string& what) {
  if (!ok) { std::printf("  FAIL: %s\n", what.c_str()); ++g_failures; }
  else     { std::printf("  ok:   %s\n", what.c_str()); }
}

// A frame header of the shape a perception pipeline actually passes around:
// POD metadata plus an index into a preallocated image pool, never the pixels.
struct FrameRef {
  std::uint64_t seq;
  std::uint64_t stamp_ns;
  std::int32_t  pool_index;
  std::int32_t  camera_id;
};

int main() {
  std::printf("SpscRing\n");

  {
    odfm::SpscRing<int> q(4);
    check(q.size_approx() == 0, "starts empty");
    int out = -1;
    check(!q.pop(out), "pop on empty returns false");

    for (std::size_t i = 0; i < q.capacity(); ++i) check(q.push(int(i)), "push until full");
    check(!q.push(999), "push on full returns false rather than blocking");
    check(q.dropped() == 1, "dropped counter increments on a full push");

    for (std::size_t i = 0; i < q.capacity(); ++i) {
      check(q.pop(out), "pop returns a value");
      check(out == int(i), "FIFO order preserved");
    }
    check(!q.pop(out), "empty again");
  }

  {
    // Wrap-around: push/pop far past the physical buffer size.
    odfm::SpscRing<int> q(8);
    int out = 0;
    bool wrap_ok = true;
    for (int i = 0; i < 1000 && wrap_ok; ++i) {
      wrap_ok = q.push(i) && q.pop(out) && out == i;
    }
    check(wrap_ok, "1000 push/pop cycles wrap correctly past the buffer size");
  }

  {
    // Real contention: one producer, one consumer, on separate threads.
    constexpr std::uint64_t kN = 1'000'000;
    odfm::SpscRing<FrameRef> q(1024);
    std::atomic<bool> done{false};
    std::uint64_t received = 0, order_errors = 0, corrupt = 0;

    std::thread producer([&] {
      for (std::uint64_t i = 0; i < kN; ++i) {
        FrameRef f{i, i * 33'333'333ull, std::int32_t(i % 64), std::int32_t(i % 6)};
        while (!q.push(f)) std::this_thread::yield();   // queue full: retry
      }
      done.store(true, std::memory_order_release);
    });

    std::thread consumer([&] {
      FrameRef f{};
      std::uint64_t expect = 0;
      while (true) {
        if (q.pop(f)) {
          if (f.seq != expect) ++order_errors;
          // Every field is derived from seq, so any tear is detectable.
          if (f.stamp_ns != f.seq * 33'333'333ull ||
              f.pool_index != std::int32_t(f.seq % 64) ||
              f.camera_id != std::int32_t(f.seq % 6)) ++corrupt;
          ++expect; ++received;
        } else if (done.load(std::memory_order_acquire) && q.size_approx() == 0) {
          break;
        }
      }
    });

    producer.join();
    consumer.join();
    check(received == kN, "no items lost across 1,000,000 handoffs");
    check(order_errors == 0, "strict FIFO under contention");
    check(corrupt == 0, "no torn values under contention");
  }

  std::printf("\nLatestFrame\n");

  {
    odfm::LatestFrame<FrameRef> slot;
    FrameRef out{};
    slot.store(FrameRef{7, 1234, 3, 5});
    check(slot.load(out), "load succeeds");
    check(out.seq == 7 && out.camera_id == 5, "value round-trips");
    check(slot.version() == 1, "version increments per store");
  }

  {
    // A writer running much hotter than the reader must never hand the reader a
    // half-written value. Every field is a function of seq, so a tear shows up.
    constexpr int kWrites = 500'000;
    odfm::LatestFrame<FrameRef> slot;
    std::atomic<bool> stop{false};
    std::uint64_t torn = 0, reads = 0, retries_exhausted = 0;

    std::thread writer([&] {
      for (std::uint64_t i = 1; i <= kWrites; ++i)
        slot.store(FrameRef{i, i * 7ull, std::int32_t(i % 64), std::int32_t(i % 6)});
      stop.store(true, std::memory_order_release);
    });

    std::thread reader([&] {
      FrameRef f{};
      while (!stop.load(std::memory_order_acquire)) {
        if (!slot.load(f)) { ++retries_exhausted; continue; }
        ++reads;
        if (f.seq == 0) continue;   // never written yet
        if (f.stamp_ns != f.seq * 7ull ||
            f.pool_index != std::int32_t(f.seq % 64) ||
            f.camera_id != std::int32_t(f.seq % 6)) ++torn;
      }
    });

    writer.join();
    reader.join();
    std::printf("  info: %llu reads, %llu retry-exhaustions\n",
                (unsigned long long)reads, (unsigned long long)retries_exhausted);
    check(torn == 0, "seqlock never exposes a torn frame under a hot writer");
    check(reads > 0, "reader made progress");
  }

  std::printf(g_failures ? "\nSpscRing/LatestFrame: %d FAILURE(S)\n"
                         : "\nSpscRing/LatestFrame: all passed\n", g_failures);
  return g_failures ? 1 : 0;
}
