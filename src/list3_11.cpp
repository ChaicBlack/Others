// This is a naive translation from a single threaded code to a
// multithreaded version but sacrifising performance because of the
// unnessicery checks for resourses availablity after the first one.
#include <mutex>
std::shared_ptr<some_resource> resource_ptr;
std::mutex resource_mutex;
void foo(){
  std::unique_lock<std::mutex> lk(resource_mutex);
  if(!resource_ptr){
    resource_ptr.reset(new some_resource);
  }
  lk.unlock();
  resource_ptr->do_something();
}
