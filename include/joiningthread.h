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
	joining_thread (joining_thread&& other) noexcept:
		t(std::move(other.t))
	{}
	// move equal operator will wait for current thread to complete if
	// it is joinable, or just move the thread in the joining thread
	// on the other side of = operator.
	joining_thread& operator=(joining_thread&& other) noexcept
	{
		if(joinable())
			join();
		t = std::move(other.t);
		// return for chain equal operator
		return *this;
	}
	// creating a new thread instead of grap a current one
	joining_thread& operator=(joining_thread other) noexcept
	{
		if(joinable())
			join();
		t = std::move(other.t);
		return *this;
	}
	~joining_thread()
	{
		if(joinable())
			join();
	}
	void swap(joining_thread& other) noexcept
	{
		t.swap(other.t);
	}
	std::thread::id get_id() const noexcept
	{
		return t.get_id();
	}
	bool joinable() const noexcept
	{
		return t.joinable();
	}
	void join()
	{
		t.join();
	}
	void detach()
	{
		t.detach();
	}
	std::thread& as_thread() noexcept
	{
		return t;
	}
	const std::thread& as_thread() const noexcept
	{
		return t;
	}
};
