// The swap operation of listing3.9 needs to lock 2 mutexes because it
// needs to operate on some_big_data. But things will change if it just
// operates on primitive types.
#include <mutex>
class Y {
private:
  int some_detail;
  mutable std::mutex m;
  int get_some_detail() const {
    std::lock_guard<std::mutex> lock_a(m);
    return this->some_detail;
  }
public:
  Y(int sd) : some_detail(sd) {}
  friend bool operator==(const Y& lhs, const Y& rhs){
    if (&lhs == &rhs)
      return true;
    // Though with many performance benifits, the data can be changed
    // between below 2 lines.
    const int lhs_value = lhs.get_some_detail();
    const int rhs_value = rhs.get_some_detail();
    return lhs_value == rhs_value;
  }
};
