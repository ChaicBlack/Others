// this is the first part of a simple atm state machine's implementation
#include <string>

class atm {

  messaging::receiver incoming;
  messaging::sender   bank;
  messaging::sender   interface_hardware;

  // 这行代码常用于状态机的实现，是一个指向成员函数的指针
  // void表示没有返回值，(atm::*state)表示一个指针，名为state，指向atm的成员函数
  // ()表示不接受任何参数
  void (atm::*state)();
  std::string account;
  std::string pin;

  void waiting_for_card(){
    interface_hardware.send(display_enter_card());
    incoming.wait().handle<card_inserted>(
        // 注意如果进来的信息不是card_inserted，会被discarded
        [&](const card_inserted & msg){
          account = msg.account;
          pin = "";
          interface_hardware.send(display_enter_pin());
          state = &atm::getting_pin;
        });
  }
  void getting_pin();
public:
  void run(){
    state = &atm::waiting_for_card;
    try {
      for(;;){
        (this->*state)();
      }
    }
    catch (messaging::close_queue const &) {}
  }
};
