// This is another way to do time consuming task, spawning a new thread

std::future<void> process_login(const std::string & username,
                                const std::string & password)
{
  return std::async(std::launch::async, [=](){
    try {
      const user_id id = backend.authenticate_user(username, password);
      const user_data info_to_display = backend.request_current_info(id);
      update_display(info_to_display);
    } catch (std::exception & e) {
      display_error(e);
    }
  });
}
