#include <thread>
#include <utility>

class some_big_project;
void swap(some_big_project &lhs, some_big_project &rhs);
class X {
private:
  some_big_project some_detail;
  std::mutex m;

public:
  X(some_big_project const &sd) : some_detail(sd) {}
  friend void swap(X &lhs, X &rhs) {
    if (&lhs == &rhs)
      return;
    std::lock(lhs.m, rhs.m);
    // std::adopt_lock is for adopting existing lock
    std::lock_guard<std::mutex> lock_a(lhs.m, std::adopt_lock);
    std::lock_guard<std::mutex> lock_b(rhs.m, std::adopt_lock);
    // After C++17, using scope_lock is a wiser choise
    // 16, 18 => written:
    // std::scope_lock lock(lhs.m, rhs.m);
    std::swap(lhs.some_detail, rhs.some_detail);
  }
};
