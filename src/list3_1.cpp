#include <algorithm>
#include <list>
#include <mutex>
std::list<int> some_list;
std::mutex some_mutex;
void add_to_list(std::list<int> some_list, int new_value) {
	// After C++17, you can omit the template argument, so
	// std::lock_guard guard(some_mutex);
	// can be written.
	// And C++17 has scope_lock.
  std::lock_guard<std::mutex> guard(some_mutex);
  some_list.push_back(new_value);
}
bool list_contains(std::list<int> some_list, int value_to_find) {
  std::lock_guard<std::mutex> guard(some_mutex);
  return std::find(some_list.begin(), some_list.end(), value_to_find) !=
         some_list.end();
}
