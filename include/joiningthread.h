// This header file describes a possible implementation of jthread
// Created by Bruce Shen, 02/04/2024



#include <thread>
#include <utility>

class joining_thread
{
	std::thread t;
public:
	joining_thread() noexcept = default;
	template<typename Callable, typename ... Args>
	explicit joining_thread(Callable&& func, Args&& ... args):
			t(std::forward<Callable>(func), std::forward<Args>(args)...)
	{}
	explicit joining_thread(std::thread t_) noexcept:
		t(std::move(t_))
	{}

}
