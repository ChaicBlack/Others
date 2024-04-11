// 这是一个线程安全的queue的接口
#include <memory>
template <typename T> class threadsafe_queue {
public:
  threadsafe_queue();
  threadsafe_queue(const threadsafe_queue &);
  threadsafe_queue &operator=(const threadsafe_queue &) = delete;
  void push(T new_value);
  bool try_pop(T &value);
  // shared_ptr的重载会在不成功的时候返回nullptr
  std::shared_ptr<T> try_pop();
  void wait_and_pop(T &value);
  std::shared_ptr<T> wait_and_pop();
  bool empty() const;
};
