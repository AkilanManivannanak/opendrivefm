// spsc_ring.hpp — lock-free primitives for a sensor -> inference pipeline.
//
// An AV perception pipeline has one producer (the camera/driver thread) and one
// consumer (the inference thread), and the producer must never block: a stalled
// sensor thread drops frames at the source, which is unrecoverable. These two
// primitives cover the two things such a pipeline actually needs.
//
//   SpscRing<T>    bounded queue, wait-free push and pop, no allocation after
//                  construction, no mutex. Use when every frame matters.
//
//   LatestFrame<T> seqlock single-slot buffer. The writer never waits; a reader
//                  that races a write retries and gets the newest complete
//                  value. Use when only the freshest frame matters, which is the
//                  common case for perception: a 40 ms old frame is worthless.
//
// Why not "overwrite the oldest" in the ring? In a pure SPSC ring only the
// consumer may advance the tail. A producer that overwrites would have to move
// the consumer's index, which races. That is why LatestFrame exists separately
// rather than as a mode of the ring.
//
// C++17, header-only, no dependencies.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <vector>

namespace odfm {

#if defined(__cpp_lib_hardware_interference_size)
inline constexpr std::size_t kCacheLine = std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t kCacheLine = 64;  // x86-64 and Apple Silicon
#endif

namespace detail {
inline std::size_t round_up_pow2(std::size_t n) {
  std::size_t p = 1;
  while (p < n) p <<= 1;
  return p;
}
}  // namespace detail

/// Bounded wait-free SPSC queue. Exactly one producer thread may call push();
/// exactly one consumer thread may call pop(). Capacity is rounded up to a power
/// of two so the modulo becomes a mask.
template <typename T>
class SpscRing {
 public:
  explicit SpscRing(std::size_t capacity)
      : mask_(detail::round_up_pow2(capacity + 1) - 1), buf_(mask_ + 1) {}

  SpscRing(const SpscRing&) = delete;
  SpscRing& operator=(const SpscRing&) = delete;

  /// Producer thread only. Returns false if the queue is full (caller decides
  /// whether to drop, which is the correct policy for a sensor thread).
  bool push(T value) {
    const std::size_t head = head_.load(std::memory_order_relaxed);
    const std::size_t next = (head + 1) & mask_;

    // Fast path uses a cached copy of the consumer index so the common case
    // touches no shared cache line at all.
    if (next == cached_tail_) {
      cached_tail_ = tail_.load(std::memory_order_acquire);
      if (next == cached_tail_) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
      }
    }
    buf_[head] = std::move(value);
    // Release: the slot write above must be visible before the index move.
    head_.store(next, std::memory_order_release);
    return true;
  }

  /// Consumer thread only. Returns false if the queue is empty.
  bool pop(T& out) {
    const std::size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == cached_head_) {
      cached_head_ = head_.load(std::memory_order_acquire);
      if (tail == cached_head_) return false;
    }
    out = std::move(buf_[tail]);
    tail_.store((tail + 1) & mask_, std::memory_order_release);
    return true;
  }

  /// Usable slots. One slot is reserved to distinguish full from empty.
  std::size_t capacity() const noexcept { return mask_; }

  /// Approximate; exact only when observed from a quiescent state.
  std::size_t size_approx() const noexcept {
    const std::size_t h = head_.load(std::memory_order_acquire);
    const std::size_t t = tail_.load(std::memory_order_acquire);
    return (h - t) & mask_;
  }

  std::uint64_t dropped() const noexcept {
    return dropped_.load(std::memory_order_relaxed);
  }

 private:
  const std::size_t mask_;
  std::vector<T> buf_;

  // head_ and tail_ live on separate cache lines: sharing one would make every
  // push invalidate the consumer's line and vice versa (false sharing), which
  // costs far more than the queue operation itself.
  alignas(kCacheLine) std::atomic<std::size_t> head_{0};
  std::size_t cached_tail_{0};   // producer-private
  alignas(kCacheLine) std::atomic<std::size_t> tail_{0};
  std::size_t cached_head_{0};   // consumer-private
  alignas(kCacheLine) std::atomic<std::uint64_t> dropped_{0};
};

/// Seqlock single-slot buffer: the writer is wait-free, readers retry while a
/// write is in flight. Restricted to trivially copyable payloads because a
/// reader may observe a partially written value before it detects the race and
/// discards it; running a non-trivial copy constructor over torn bytes is
/// undefined behaviour.
template <typename T>
class LatestFrame {
  static_assert(std::is_trivially_copyable<T>::value,
                "LatestFrame requires a trivially copyable payload (POD frame "
                "header, pointer, or index into a preallocated pool).");

 public:
  /// Writer thread only. Never blocks, never fails.
  void store(const T& value) {
    const std::uint64_t s = seq_.load(std::memory_order_relaxed);
    seq_.store(s + 1, std::memory_order_release);        // odd: write in flight
    std::atomic_thread_fence(std::memory_order_release);
    std::memcpy(&slot_, &value, sizeof(T));
    seq_.store(s + 2, std::memory_order_release);        // even: complete
  }

  /// Reader threads. Returns false only if `max_retries` races occur, which
  /// means the writer is running far hotter than the reader.
  bool load(T& out, int max_retries = 128) const {
    for (int i = 0; i < max_retries; ++i) {
      const std::uint64_t before = seq_.load(std::memory_order_acquire);
      if (before & 1u) continue;                         // write in flight
      std::memcpy(&out, &slot_, sizeof(T));
      std::atomic_thread_fence(std::memory_order_acquire);
      if (seq_.load(std::memory_order_relaxed) == before) return true;
    }
    return false;
  }

  std::uint64_t version() const noexcept {
    return seq_.load(std::memory_order_acquire) / 2;
  }

 private:
  alignas(kCacheLine) mutable std::atomic<std::uint64_t> seq_{0};
  alignas(kCacheLine) T slot_{};
};

}  // namespace odfm
