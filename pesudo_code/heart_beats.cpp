#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <atomic>

// 模拟节点结构体
struct Node {
    int id;
    bool is_alive;
};

// 发送ping请求的函数
bool ping(Node& node) {
    // 模拟网络延迟和节点响应
    std::this_thread::sleep_for(std::chrono::milliseconds(rand() % 1000));
    return node.is_alive;
}

// 广播ping的函数
void broadcast_ping(std::vector<Node>& nodes, std::unordered_map<int, bool>& crashed_nodes) {
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<int> active_pings(0);

    for (auto& node : nodes) {
        active_pings++;
        threads.emplace_back([&node, &crashed_nodes, &mutex, &cv, &active_pings]() {
            bool response = ping(node);
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!response) {
                    crashed_nodes[node.id] = true;
                }
            }
            active_pings--;
            cv.notify_one();
        });
    }

    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::milliseconds(500), [&active_pings] { return active_pings == 0; });
    }

    // 标记没有响应的节点为crashed_node
    for (auto& node : nodes) {
        if (crashed_nodes.find(node.id) == crashed_nodes.end()) {
            crashed_nodes[node.id] = false;
        }
    }
}

int main() {
    // 创建一些节点
    std::vector<Node> nodes = {
        {1, true},
        {2, false},  // 模拟一个crashed_node
        {3, true},
        {4, false},  // 模拟另一个crashed_node
        {5, true}
    };

    // 存储crashed_node
    std::unordered_map<int, bool> crashed_nodes;

    // 进行广播ping
    broadcast_ping(nodes, crashed_nodes);

    // 打印crashed_node
    for (const auto& node_status : crashed_nodes) {
        if (node_status.second) {
            std::cout << "Node " << node_status.first << " is crashed." << std::endl;
        }
    }

    return 0;
}

