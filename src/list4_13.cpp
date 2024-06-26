// this is a FP style list_quick_sort implementation
// the feature of FP makes it easy to be parallelized
// a advantage is that compiler would choose to ran synchronized if
// there are too many threads existing.
#include <algorithm>
#include <future>
#include <list>
template <typename T> std::list<T> parallel_quick_sort(std::list<T> input) {
  if (input.empty()) {
    return input;
  }
  std::list<T> result;
  result.splice(result.begin(), input, input.begin());
  const T &pivot = *result.begin();
  auto divide_point = std::partition(input.begin(), input.end(),
                                     [&](const T &t) { return t < pivot; });
  std::list<T> lower_part;
  lower_part.splice(lower_part.end(), input, input.begin(), divide_point);
  std::future<std::list<T>> new_lower =
      std::async(parallel_quick_sort, std::move(lower_part));
  auto new_higher(parallel_quick_sort(std::move(input)));
  result.splice(result.end(), new_higher);
  result.splice(result.begin(), new_lower.get());
}
