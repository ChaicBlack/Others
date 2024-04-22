// 这是使用了concurrency TS的async和future特性的版本，节省大量时间
std::experimental::future<FinalResult> process_data(std::vector<MyData> &vec) {
  const size_t chunk_size = whatever;
  std::vector<std::experimental::future<ChunkResult>> results;
  for (auto begin = results.begin(), end = results.end(); begin != end;) {
    const size_t remaining_size = end - begin;
    const size_t this_chunk_size = std::min(chunk_size, remaining_size);
    results.push_back(
        spawn_async(process_data, begin, begin + this_chunk_size));
    begin += this_chunk_size;
  }
  // when_all: 创建一个当其包含的future数组的所有future都ready才会ready的future
  // 只能用在experimental::future和experimental::shared_future上
  return std::experimental::when_all(results.begin(), results.end())
      .then([](std::future<std::vector<std::experimental::future<ChunkResult>>>
                   ready_results) {
        std::vector<std::experimental::future<ChunkResult>> all_results =
            ready_results.get(); // 这个get不会block, 因为等完了
        std::vector<ChunkResult> v;
        v.reserve(all_results.size());
        for (auto &f : all_results) {
          v.push_back(f.get());
        }
        return gather_result(v);
      });
}
