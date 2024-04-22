auto fut = spawn_async(some_function).share();
auto fut2 = fut.then([](std::experimental::shared_future<some_data> data){
  do_stuff(data);
});
auto fut3 = fut.then([](std::experimental::shared_future<some_data> data){
  do_other_stuff(data);
});
// fut2 和 fut3 都是future而不是shared_future
// 这段代码展示了shared_future的用处，也就是可以使用多个continuation，不论是在
// 单个instance上还是在多个instance上
