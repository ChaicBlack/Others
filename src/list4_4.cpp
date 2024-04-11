// 这是一个对threadsafe_queue的应用
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T> class threadsafe_queue {
private:
  std::mutex thread_mutex;
  std::condition_variable condition;
  std::queue<T> data_queue;

public:
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
};
threadsafe_queue<data_type> data_que;
void data_prepare_thread() {
  while (more_data_to_prepare()) {
    data_type data = prepare_data();
    data_que.push(data);
  }
}
void data_process_thread() {
  while (true) {
    data_type data = data_que.wait_and_pop();
    process(data);
    if (is_last_data())
      break;
  }
}
