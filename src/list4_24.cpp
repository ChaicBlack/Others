// Using std::experimental::when_any to process first value found

std::experimental::future<FinalResult>
find_and_process_value(std::vector<Mydata &> data) {
  // 这个函数是用来获取硬件支持的线程数的，0就是没定义或者计算不了
  const unsigned concurrency = std::thread::hardware_concurrency();
  const unsigned num_tasks = (concurrency > 0) ? concurrency : 2;
  std::vector<std::experimental::future<MyData *>> results;
  const auto chunk_size = (data.size() + num_tasks - 1) / num_tasks;
  auto chunk_begin = data.begin();
  std::shared_ptr<std::atomic<bool>> done_flag =
      std::make_shared<std::atomic<bool>>(false);
  for (unsigned i = 0; i < num_tasks; i++) {
    auto chunk_end =
        (i < (num_tasks - 1)) ? chunk_begin() + chunk_size : data.end();
    results.push_back(spawn_async([=] {
      for (auto entry = chunk_begin; !*done_flag && (entry != chunk_end);
           entry++) {
        if (matches_find_criteria(*entry)) {
          *done_flag = true;
          return &*entry; // 将迭代器转变成指针
        }
      }
      return (MyData *)nullptr;
    }));
    chunk_begin = chunk_end;
  }

  std::shared_ptr<std::experimental::promise<FinalResult>> final_result =
      std::make_shared<std::experimental::promise<FinalResult>>();

  // 这个DoneCheck定义成class就是为了重复调用回调函数的
  struct DoneCheck {
    std::shared_ptr<std::experimental::promise<FinalResult>> final_result;
    DoneCheck(
        std::shared_ptr<std::experimental::promise<FinalResult>> final_result_)
        : final_result(std::move(final_result_)) {}
    void
    operator()(std::experimental::future<std::experimental::when_any_result<
                   std::vector<std::experimental::future<MyData *>>>>
                   result_param) {
      auto results = result_param.get();
      const MyData *ready_result = results.futures[results.index].get();
      if (ready_result)
        final_result->set_value(process_found_value(*ready_result));
      else {
        results.futures.erase(results.futures.begin(), results.index);
        if (!results.futures.empty()) {
          // 重新触发回调函数，并使用DoneCheck()作为回调(then接受callable)
          std::experimental::when_any(results.futures.begin(),
                                      results.futures.end())
              .then(std::move(*this));
        } else {
          final_result->set_exception(
              std::make_exception_ptr(std::runtime_error("Not found")));
        }
      }
    }
  };

  std::experimental::when_any(results.begin(), results.end())
      .then(DoneCheck(final_result));
  return final_result->get_future();
}
