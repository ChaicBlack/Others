// 这是没有predicate的wait_until一般使用方式。
// 必须要使用loop来应对虚假唤醒。
// 而wait_for则不能loop，因为每次循环都会重制计时器，让等待时间无限拉长
#include <condition_variable>
#include <mutex>
#include <chrono>

std::mutex m;
std::condition_variable cv;
bool done;
bool wait_loop(){
  auto const time_out = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(500);
  std::unique_lock<std::mutex> lk(m);
  while(!done){
    if(cv.wait_until(lk, time_out) == std::cv_status::timeout)
      break;
  }
  return done;
}
