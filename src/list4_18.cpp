// This is a simple sequential function to process user login

void process_login(const std::string & username, const std::string & password)
{
  try {
    const user_id id = backend.authenticate_user(username, password);
    const user_data info_to_display = backend.request_current_info(id); 
  } catch(std::exception & e) {
    display_error(e);
  }
}
