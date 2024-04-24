// waiting for events with std::experimental::latch
void foo() {
  const unsigned thread_count = ...;
  latch done(thread_count);
  my_data data[thread_count];
  std::vector<std::future<void>> threads;
  for (int i = 0; i < thread_count; i++) {
    // 除了i都捕获引用，为防止data race所以不能捕获i
    threads.push_back(std::async(std::launch::async, [&, i] {
      data[i] = make_data(i);
      done.count_down();
      do_more_stuff();
    }));
  }
  done.wait();
  process_data(data, thread_count);
}
