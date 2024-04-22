// 这是一个在不使用一对多future的情况下处理大数据的一个implementation
// 对每一chunk数据都async出去处理，最后还要单独async一个收集的线程
std::future<FinalResult> process_data(std::vector<MyData> & vec){
  const size_t chunk_size = whatever;
  std::vector<std::future<ChunkResult>> results;
  // async任务出去，并返回future加入向量
  for(auto begin = vec.begin(), end = vec.end(); begin != end;){
    const size_t remainning_size = end - begin;
    const size_t this_chunk_size = std::min(remainning_size, chunk_size);
    results.push_back(std::async(process_chunk, begin, begin + this_chunk_size));
    begin += this_chunk_size;
  }
  // async另一个线程专门等待所有的future处理完毕，得到结果
  return std::async([all_results=std::move(results)](){
    std::vector<ChunkResult> v;
    v.reserve(all_results.size());
    for(auto & f : all_results){
      // 不仅要等待，而且有context switching的overhead
      v.push_back(f.get());
    }
    return gather_results(v);
  });
}
