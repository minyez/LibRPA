//=======================
// AUTHOR : Peize Lin
// DATE :   2026-04-21
//=======================

#pragma once

#include "../global/Global_Func.h"
#include "State.h"

#include <vector>
#include <atomic>
#include <algorithm>

namespace Comm
{

class Memory_Check
{
  public:
	bool first_check_send = false;
	bool first_check_recv = false;
	std::atomic<bool> first_set_send{false};
	std::atomic<bool> first_set_recv{false};
	std::atomic<std::size_t> max_used_send{0};
	std::atomic<std::size_t> max_used_recv{0};
	const std::vector<std::atomic<State_Send>> &states_send;
	const std::vector<std::atomic<State_Recv>> &states_recv;

	Memory_Check(
		const std::vector<std::atomic<State_Send>> &states_send_in,
		const std::vector<std::atomic<State_Recv>> &states_recv_in
	): states_send(states_send_in), states_recv(states_recv_in){}


	std::size_t get_memory_ar() const
	{
		return this->max_used_send * std::count(std::begin(this->states_send), std::end(this->states_send), State_Send::begin_oar)
		     + this->max_used_recv * std::count(std::begin(this->states_recv), std::end(this->states_recv), State_Recv::begin_iar);
	}

	void set_max_used_send(const std::size_t used)
	{
		this->max_used_send = std::max(used, this->max_used_send.load());
		this->first_set_send = true;
	}

	void set_max_used_recv(const std::size_t used)
	{
		this->max_used_recv = std::max(used, this->max_used_recv.load());
		this->first_set_recv = true;
	}

	bool enough_send()
	{
		if(!this->first_check_send)
		{
			this->first_check_send = true;
			return true;
		}
		else if(!this->first_set_send)
		{
			return false;
		}
		else
		{
			return (this->max_used_send.load() + this->get_memory_ar())
				< Global_Func::memory_available();
		}
	}

	bool enough_recv()
	{
		if(!this->first_check_recv)
		{
			this->first_check_recv = true;
			return true;
		}
		else if(!this->first_set_recv)
		{
			return false;
		}
		else
		{
			return (this->max_used_recv.load() * 2 + this->get_memory_ar())
				< Global_Func::memory_available();
		}
	}
};

}