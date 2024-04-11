// 这是一个完整的threadsafe_queue的实现
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

template <typename T> class threadsafe_queue {
private:
  // mutable表示的是在声明为const的成员函数或者const对象中仍然可以对其进行修改
  // 就比如empty和copy constructor需要上锁同时需要承诺不修改其他成员变量
  mutable std::mutex thread_mutex;
  std::condition_variable condition;
  std::queue<T> data_queue;

public:
  threadsafe_queue() = default;
  threadsafe_queue(const threadsafe_queue &other) {
    std::lock_guard<std::mutex> lk(thread_mutex);
    data_queue = other.data_queue;
  }
  void push(const T &data) {
    std::lock_guard<std::mutex> lk(thread_mutex);
    data_queue.push(data);
    condition.notify_one();
  }
  void wait_and_pop(T &data) {
    std::unique_lock<std::mutex> lk(thread_mutex);
    condition.wait(lk, [this] { return !data_queue.empty(); });
    data = data_queue.front();
    data_queue.pop();
  }
  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lk(thread_mutex);
    condition.wait(lk, [this] { return !data_queue.empty(); });
    auto res(std::make_shared(data_queue.front()));
    return res;
  }
  bool try_pop(T &data) {
    std::lock_guard<std::mutex> lk(thread_mutex);
    if (data_queue.empty())
      return false;
    data = data_queue.front();
    data_queue.pop();
    return true;
  }
  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lk(thread_mutex);
    if (data_queue.empty())
      return std::shared_ptr<T>();
    // 伟大的类型推断
    auto res(std::make_shared(data_queue.front()));
    data_queue.pop();
    return res;
  }
  bool empty() const {
    std::lock_guard<std::mutex> lk(thread_mutex);
    return data_queue.empty();
  }
};
