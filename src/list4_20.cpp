// Using continuation to solve previous problem

std::experimental::future<void> process_login(const std::string & username,
                                              const std::string & password)
{
  return spawn_async([=](){
    return backend.authenticate_user(username, password);
  }).then([](std::experimental::future<user_id> id){
    return backend.request_current_info(id.get());
  }).then([](std::experimental::future<user_data> info_to_display){
    try {
      update_display(info_to_display.get());
    } catch (std::exception & e) {
      // All the exception will propagated all the way down the chain
      display_error(e);
    }
  });
}
