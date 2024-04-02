#include <fmt/core.h>

#include <thread>
#include <vector>

int main() {
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < 20; i++) {
    threads.emplace_back([i]() { fmt::print("{}\n", i); });
  }
  for (auto &entry : threads) entry.join();
  return 0;
}
