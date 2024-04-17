// 这是上一个代码里的getting_pin状态函数的implementation
void atm::getting_pin() {
  // 3个handle串联，wait可以处理三种信息
  incoming.wait()
      .handle<digit_pressed>([&](const digit_pressed &msg) {
        const unsigned pin_length = 4;
        pin += msg.digit;
        if (pin.length() == pin_length) {
          bank.send(verify_pin(account, pin, incoming));
          state = &atm::verifying_pin;
        }
      })
      .handle<clear_last_pressed>([&](const clear_last_pressed &msg) {
        if (!pin_empty()) {
          pin.resize(pin.length() - 1);
        }
      })
      .handle<cancel_pressed>(
          [&](const cancel_pressed &msg) { state = &atm::done_processing; });
}
