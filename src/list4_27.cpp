// using std::exprtimental::flex_barrier
void process_data(data_source &source, data_sink &sink) {
  const unsigned concurrency = std::thread::hardware_concurrency();
  const unsigned num_threads = (concurrency > 0) ? concurrency : 2;
  std::vector<data_chunk> chunks;
  auto split_source = [&] {
    if (!source.done()) {
      data_block current_block = source.get_next_data_block();
      chunks = divide_into_chunks(current_block, num_threads);
    }
  };

  split_source();

  result_block result;

  std::experimental::flex_barrier sync(num_threads, [&] {
    sink.write_data(std::move(result));
    split_source();
    // -1意味着线程数不改变，大于-1那么下一次循环线程数就是这个返回值
    return -1;
  });

  std::vector<joining_thread> threads;

  for (unsigned i = 0; i < num_threads; i++) {
    threads[i] = joining_threads([&, i] {
      while (!source.done()) {
        result.set_chunk(i, num_threads, process(chunks[i]));
        sync.arrive_and_wait();
      }
    });
  }
}
