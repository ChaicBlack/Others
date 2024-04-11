#include <condition_variable>
#include <mutex>
#include <queue>

std::mutex mut;
std::queue<data_chunk> data_queue;
std::condition_variable condition;

void prepare_data_thread() {
  data_chunk data = make_parepared_data();
  {
    std::unique_lock<std::mutex> lk(mut);
    data_queue.push(data);
  }
  condition.notify_one();
}

void data_process_thread() {
  while (true) {
    std::unique_lock<std::mutex> lk(mut);
    condition.wait(lk, [] { return !data_queue.empty(); });
    data_chunk data = data_queue.front();
    data_queue.pop();
    process_data(data);
    if (is_last_data())
      break;
  }
}

/* 一个可能的std::condition_vatiable::wait的implementation:
 * template<typename Predicate>
 * void minimal_wait(std::unique_lock & lk, Predicate pred)
 * {
 *    while(!pred()){
 *      lk.unlock();
 *      lk.lock();
 *    }
 * }
 * 你的代码必须准备好与这种简陋的实现来合作*/
