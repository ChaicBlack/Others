#include "list3_8.cpp"

hierarchical_mutex high_level_mutex(10000);

hierarchical_mutex low_level_mutex(5000);

hierarchical_mutex other_mutex(6000);

int do_low_level_stuff();

int low_level_func() {
  std::lock_guard<hierarchical_mutex> lk(low_level_mutex);
  return do_low_level_stuff();
}

void high_level_stuff(int some_param);

void high_level_func() {
  std::lock_guard<hierarchical_mutex> lk(high_level_mutex);
  high_level_stuff(low_level_func());
}

void thread_a() { high_level_func(); }

void do_other_stuff();

void other_stuff() {
  high_level_func();
  do_other_stuff();
}

void thread_b() {
  std::lock_guard<hierarchical_mutex> lk(other_mutex);
  other_stuff();
}
