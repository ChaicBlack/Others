// listing3.6 can easily be rewritten to this by using std::unique_lock-
// and std::defer_lock rather than std::lock_guard and std::adopt_lock
#include <mutex>

class some_big_object {};
void swap(some_big_object &, some_big_object &);
class X {
private:
  some_big_object some_detail;
  std::mutex m;
public:
  X(const some_big_object & sd) : some_detail(sd) {}
  friend void swap(X & lhs, X & rhs)
  {
    if(&lhs == &rhs)
      return;
    std::unique_lock<std::mutex> lock_a(lhs.m, std::defer_lock);
    std::unique_lock<std::mutex> lock_b(rhs.m, std::defer_lock);
    std::lock(lock_a, lock_b);
    swap(lhs, rhs);
  }
};
