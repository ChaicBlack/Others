// 这是一个使用std::atomic_flag实现的自旋锁
// 它满足了std::lock_guard<>对锁的要求

#include <atomic>
class spinlock_mutex {
  std::atomic_flag flag;
public:
  spinlock_mutex() : flag(ATOMIC_FLAG_INIT) {}
  void lock() {
    while(flag.test_and_set(std::memory_order_acquire));
  }
  void unlock() {
    flag.clear(std::memory_order_release);
  }
};
