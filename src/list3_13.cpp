// 这是一个简单的shared_mutex和shared_lock的实践，不多赘述
// 越来越能体会到rust的设计用心了
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>

class dns_entry {};
class dns_cache {
private:
  mutable std::shared_mutex entry_mutex;
  std::map<std::string, dns_entry> entries;

public:
  dns_entry find_entry(const std::string &domain) const {
    std::shared_lock<std::shared_mutex> lk(entry_mutex);
    auto it = entries.find(domain);
    return (it != entries.end()) ? it->second : dns_entry();
  }
  void update_or_add_entry(const std::string &domain,
                           const dns_entry &dns_detail) {
    std::lock_guard lk(entry_mutex);
    entries[domain] = dns_detail;
  }
};
