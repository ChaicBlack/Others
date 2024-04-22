// This is a fully asynchronous operation of previous list
std::experimental::future<void> process_login(const std::string &username,
                                              const std::string &password) {
  return backend.async_authenticate_user(username, password)
      // c++14以后可以换成auto id，大大节省代码量
      .then([](std::experimental::future<user_id> id) {
        return backend.async_request_current_info(id.get());
      })
      .then([](std::experimental::future<user_data> info_to_display) {
        try {
          update_display(info_to_display.get());
        } catch (std::exception &e) {
          display_error(e);
        }
      });
}
// 其实这些例子都不足以说明continuation的威力，因为代码量不够多
// 在每一个任务分割的线程都可能存在着其他任务，如果不使用then这种
// 的话，就会导致大量的等待和blocking，浪费大量线程的时间和资源。
// 使用then，就可以去执行其他任务。
