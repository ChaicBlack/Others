// This list illustrated the std::async and std::future, both defined in
// the <future> header.
#include <fmt/core.h>
#include <future>

int find_the_result_of_ltuae();
void do_something_else();
int main() {
  // lutae, 生命\宇宙\还有一切
  std::future<int> answer_of_ltuae = std::async(find_the_result_of_ltuae);
  do_something_else();
  fmt::print("{}\n", answer_of_ltuae.get());
}
