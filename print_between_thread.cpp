#include <mutex>
#include <thread>
#include <fmt/core.h>
#include <condition_variable>

std::mutex mtx;
bool printA;
std::condition_variable cv;
void interprint(char ch, const int count){
  for(int i = 0; i < count; i++){
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, [ch]{ return (ch == 'A' && printA) || (ch == 'B' && !printA); });
    fmt::print("{}\n", ch);
    printA = !printA;
    cv.notify_one();
  }
}

int main(){
  printA = true;
  const int COUNT = 10;
  std::thread t1(interprint, 'A', COUNT);
  std::thread t2(interprint, 'B', COUNT);
  t1.join();
  t2.join();
  return 0;
}
