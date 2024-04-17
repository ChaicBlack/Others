// This is a possible spawn_task implementation which is like async
#include <future>
#include <thread>
#include <utility>
template <typename F, typename... Args>
std::future<typename std::invoke_result<F(Args &&...)>::type>
spawn_task(F &&f, Args &&...args) {
  using result_type = std::invoke_result<F(Args && ...)>::type;
  std::packaged_task<result_type(Args && ...)> task(std::forward<F>(f));
  std::future<result_type> res = task.get_future();
  std::thread t(std::move(task), std::forward<Args>(args)...);
  t.detach();
  return res;
}
