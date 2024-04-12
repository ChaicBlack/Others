// This listing is mainly talk about the different use of RAII, which
// includes std::thread, std::bind, std::ref, std::async and other stuff
#include <future>
#include <string>

struct X {
  void foo(int, const std::string &);
  std::string bar(const std::string &);
};
X x;
auto f1 = std::async(&X::foo, &x, 42, "hello");
auto f2 = std::async(&X::bar, x, "goodbay");

struct Y {
  double operator()(double);
};
Y y;
auto f3 = std::async(Y(), 3.141);
auto f4 = std::async(std::ref(y), 2.712);
X baz(X &);
std::async(baz, std::ref(x));

struct move_only {
  move_only();
  move_only(move_only &&);
  move_only(const move_only &) = delete;
  move_only &operator=(move_only &&);
  move_only &operator=(const move_only &) = delete;
  void operator()();
};
auto f5 = std::async(move_only());

// std::launch的用法
auto f6 = std::async(std::launch::async, Y(), 1.2); // run in a new thread
// run in wait() or get()
auto f7 = std::async(std::launch::deferred, baz, std::ref(x));
auto f8 = std::async(std::launch::deferred | std::launch::async, baz,
                     std::ref(x));      // implementation决定
auto f9 = std::async(baz, std::ref(x)); // 同f8，因为默认implementation决定

f7.wait(); // f7被推迟到现在才开始
