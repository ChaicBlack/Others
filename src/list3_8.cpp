#include <mutex>
#include <stdexcept>

class hierarchical_mutex {
  std::mutex internal_mutex;
  unsigned long const hierarchy_value;
  unsigned long previous_hierarchy_value;
  // This is like a upper limit of the mutex current thread can lock.
  static thread_local unsigned long this_thread_hierarchy_value;
  void check_for_hierarchy_violation() {
    if (this_thread_hierarchy_value <= hierarchy_value)
      throw std::logic_error("mutex hierarchy violated");
  }
  void update_hierarchy_value() {
    previous_hierarchy_value = this_thread_hierarchy_value;
    this_thread_hierarchy_value = hierarchy_value;
  }

public:
  explicit hierarchical_mutex(unsigned long value)
      : hierarchy_value(value), previous_hierarchy_value(0) {}
  void lock() {
    check_for_hierarchy_violation();
    internal_mutex.lock();
    update_hierarchy_value();
  }
  void unlock() {
    // The unlock order is the reverse order, under what the higher
    // mutex need be unlock later for not confuse the hierarchy.
    if (this_thread_hierarchy_value != hierarchy_value)
      throw std::logic_error("mutex hierarchy violated");
    // It seems like this implementation can only store 2 hierarchy value.
    // Needing a vector to store previous locks' value list.
    // Got it, one mutex need only restore the adjasant previous one.
    this_thread_hierarchy_value = previous_hierarchy_value;
    internal_mutex.unlock();
  }
  bool try_lock() {
    check_for_hierarchy_violation();
    if (!internal_mutex.try_lock()) {
      return false;
    }
    update_hierarchy_value();
    return true;
  }
};
thread_local unsigned long
    hierarchical_mutex::this_thread_hierarchy_value(ULONG_MAX);
