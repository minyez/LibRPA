//=======================
// AUTHOR : Peize Lin
// DATE :   2022-01-05
//=======================

#pragma once

#include "Comm_Trans.h"
#include "Memory_Check.h"

#include <vector>
#include <queue>
#include <string>
#include <stdexcept>
#include <cassert>

#include <cereal/archives/binary.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/map.hpp>

#define MPI_CHECK(x) if((x)!=MPI_SUCCESS)	throw std::runtime_error(std::string(__FILE__)+" line "+std::to_string(__LINE__));

namespace Comm
{

template<typename Tkey, typename Tvalue, typename Tdatas_isend, typename Tdatas_recv>
Comm_Trans<Tkey,Tvalue,Tdatas_isend,Tdatas_recv>::Comm_Trans(const MPI_Comm &mpi_comm_in)
	:mpi_comm(mpi_comm_in)
{
	MPI_CHECK (MPI_Comm_size (this->mpi_comm, &this->comm_size));
	MPI_CHECK (MPI_Comm_rank (this->mpi_comm, &this->rank_mine));

	this->set_value_recv
		= [](Tkey &&key, Tvalue &&value, Tdatas_recv &datas_recv)
		{ throw std::logic_error("Function set_value not set."); };
	this->traverse_isend
		= [](const Tdatas_isend &datas_isend, const int rank_isend, std::function<void(const Tkey&, const Tvalue&)> &func)
		{ throw std::logic_error("Function traverse not set."); };
	this->init_datas_local
		= [](const int rank_recv) -> Tdatas_recv
		{ throw std::logic_error("Function init_datas_local not set."); };
	this->add_datas
		= [](Tdatas_recv &&datas_local, Tdatas_recv &datas_recv)
		{ throw std::logic_error("Function add_datas not set."); };
}


/*
template<typename Tkey, typename Tvalue, typename Tdatas_isend, typename Tdatas_recv>
Comm_Trans<Tkey,Tvalue,Tdatas_isend,Tdatas_recv>::Comm_Trans(const Comm_Trans &com)
	:mpi_comm(com.mpi_comm)
{
	//ofs<<"C"<<" ";
	MPI_CHECK (MPI_Comm_size (this->mpi_comm, &this->comm_size));
	MPI_CHECK (MPI_Comm_rank (this->mpi_comm, &this->rank_mine));
	this->set_value_recv = com.set_value_recv;
	this->traverse_isend = com.traverse_isend;
	this->flag_lock_set_value = com.flag_lock_set_value;
	this->init_datas_local = com.init_datas_local;
	this->add_datas = com.add_datas;
}
*/


template<typename Tkey, typename Tvalue, typename Tdatas_isend, typename Tdatas_recv>
void Comm_Trans<Tkey,Tvalue,Tdatas_isend,Tdatas_recv>::communicate(
	const Tdatas_isend &datas_isend,
	Tdatas_recv &datas_recv)
{
	std::vector<MPI_Request> requests_isend(this->comm_size);
	std::vector<std::string> buffers_isend(this->comm_size);
	std::vector<std::future<std::size_t>> futures_oar(this->comm_size);
	std::vector<std::vector<char>> buffers_recv(this->comm_size);
	std::queue<std::pair<MPI_Status,MPI_Message>> status_message_s_recv;
	std::vector<std::atomic<State_Send>> states_send(this->comm_size);
	for(std::atomic<State_Send> & state_send : states_send)
		state_send = State_Send::unstart;
	std::vector<std::atomic<State_Recv>> states_recv(this->comm_size);
	for(std::atomic<State_Recv> & state_recv : states_recv)
		state_recv = State_Recv::unstart;
	std::atomic_flag lock_set_value = ATOMIC_FLAG_INIT;
	Memory_Check memory(states_send, states_recv);

	// initialization
	int rank_isend_tmp1=1, rank_isend_tmp2=1, rank_isend_tmp3=1;

	while (!check_finish(states_send, states_recv))
	{
		while(true)
		{
			int flag_iprobe=0;
			MPI_Status status_recv;
			MPI_Message message_recv;
			MPI_CHECK (MPI_Improbe(MPI_ANY_SOURCE, this->tag_data, this->mpi_comm, &flag_iprobe, &message_recv, &status_recv));
			if (flag_iprobe)
				status_message_s_recv.emplace(status_recv, message_recv);
			else
				break;
		}

		if (!status_message_s_recv.empty() && memory.enough_recv())
		{
			const MPI_Status status_recv = status_message_s_recv.front().first;
			const MPI_Message message_recv = status_message_s_recv.front().second;
			const int rank_recv = status_recv.MPI_SOURCE;
			status_message_s_recv.pop();

			this->recv_data(
				status_recv,
				message_recv,
				memory,
				buffers_recv[rank_recv],
				states_recv[rank_recv]);

			std::async (std::launch::async,
				&Comm_Trans::iar_data, this,
					rank_recv,
					std::ref(buffers_recv[rank_recv]),
					std::ref(lock_set_value),
					std::ref(datas_recv),
					std::ref(states_recv[rank_recv])).wait();
		}

		if (rank_isend_tmp1<this->comm_size+1 && memory.enough_send())
		{
			const int rank_isend = (rank_isend_tmp1 + this->rank_mine) % this->comm_size;
			futures_oar[rank_isend] = std::async (std::launch::async,
				&Comm_Trans::oar_data, this,
					rank_isend,
					std::cref(datas_isend),
					std::ref(buffers_isend[rank_isend]),
					std::ref(states_send[rank_isend]),
					std::ref(memory));
			++rank_isend_tmp1;
		}

		while(rank_isend_tmp2<rank_isend_tmp1)
		{
			const int rank_isend = (rank_isend_tmp2 + this->rank_mine) % this->comm_size;
			if(futures_oar[rank_isend].valid())
			{
				const std::size_t exponent_align = futures_oar[rank_isend].get();
				this->isend_data(
					rank_isend,
					exponent_align,
					buffers_isend[rank_isend],
					requests_isend[rank_isend],
					states_send[rank_isend]);
				++rank_isend_tmp2;
			}
		}

		while(rank_isend_tmp3<rank_isend_tmp2)
		{
			const int rank_isend = (rank_isend_tmp3 + this->rank_mine) % this->comm_size;
			if(states_send[rank_isend] == State_Send::begin_isend)
			{
				int flag_finish=0;
				MPI_CHECK (MPI_Test (&(requests_isend[rank_isend]), &flag_finish, MPI_STATUS_IGNORE));
				if (flag_finish)
				{
					//MPI_CHECK (MPI_Request_free (&requests_isend[rank_isend]));
					buffers_isend[rank_isend].clear();
					buffers_isend[rank_isend].shrink_to_fit();
					states_send[rank_isend] = State_Send::finish_isend;
					++rank_isend_tmp3;
				}
				else {break;}
			}
			else {break;}
		}
	}
}


template<typename Tkey, typename Tvalue, typename Tdatas_isend, typename Tdatas_recv>
std::size_t Comm_Trans<Tkey,Tvalue,Tdatas_isend,Tdatas_recv>::oar_data(
	const int rank_isend,
	const Tdatas_isend &datas_isend,
	std::string &buffer_isend,
	std::atomic<State_Send> &state_send,
	Memory_Check &memory)
{
	assert(state_send == State_Send::unstart);
	state_send = State_Send::begin_oar;
	std::stringstream ss_isend;
	{
		cereal::BinaryOutputArchive oar(ss_isend);

		size_t size_item = 0;
		oar(size_item);					// 占位

		std::function<void(const Tkey&, const Tvalue&)> archive_data = [&oar, &size_item](
			const Tkey &key, const Tvalue &value)
		{
			oar(key, value);
			++size_item;
		};
		this->traverse_isend(datas_isend, rank_isend, archive_data);

		ss_isend.rdbuf()->pubseekpos(0);		// 返回size_item的占位，序列化真正的size_item值
		oar(size_item);
	} // end cereal::BinaryOutputArchive
	const std::size_t exponent_align = this->cereal_func.align_stringstream(ss_isend);
	buffer_isend = std::move(ss_isend.str());
	memory.set_max_used_send(buffer_isend.size()*sizeof(char));
	state_send = State_Send::finish_oar;
	return exponent_align;
}


template<typename Tkey, typename Tvalue, typename Tdatas_isend, typename Tdatas_recv>
void Comm_Trans<Tkey,Tvalue,Tdatas_isend,Tdatas_recv>::isend_data(
	const int rank_isend,
	const std::size_t exponent_align,
	std::string &buffer_isend,
	MPI_Request &request_isend,
	std::atomic<State_Send> &state_send)
{
	assert(state_send == State_Send::finish_oar);
	this->cereal_func.mpi_isend(buffer_isend, exponent_align, rank_isend, this->tag_data, this->mpi_comm, request_isend);
	state_send = State_Send::begin_isend;
}


template<typename Tkey, typename Tvalue, typename Tdatas_isend, typename Tdatas_recv>
void Comm_Trans<Tkey,Tvalue,Tdatas_isend,Tdatas_recv>::recv_data (
	const MPI_Status status_recv,
	const MPI_Message message_recv,
	Memory_Check &memory,
	std::vector<char> &buffer_recv,
	std::atomic<State_Recv> &state_recv)
{
	assert(state_recv == State_Recv::unstart);
	state_recv = State_Recv::begin_recv;
	MPI_Message message_recv_tmp = message_recv;
	buffer_recv = this->cereal_func.mpi_mrecv(message_recv_tmp, status_recv);
	assert(message_recv_tmp == MPI_MESSAGE_NULL);
	memory.set_max_used_recv(buffer_recv.size()*sizeof(char));
	state_recv = State_Recv::finish_recv;
}


template<typename Tkey, typename Tvalue, typename Tdatas_isend, typename Tdatas_recv>
void Comm_Trans<Tkey,Tvalue,Tdatas_isend,Tdatas_recv>::iar_data (
	const int rank_recv,
	std::vector<char> &buffer_recv,
	std::atomic_flag &lock_set_value,
	Tdatas_recv &datas_recv,
	std::atomic<State_Recv> &state_recv) const
{
	assert(state_recv == State_Recv::finish_recv);
	state_recv = State_Recv::begin_iar;
	std::stringstream ss_recv;
	ss_recv.rdbuf()->pubsetbuf(buffer_recv.data(), buffer_recv.size());
	{
		cereal::BinaryInputArchive iar(ss_recv);
		size_t size_item;	iar(size_item);

		if (this->flag_lock_set_value==Comm_Tools::Lock_Type::Lock_free)
		{
			for (size_t i=0; i<size_item; ++i)
			{
				Tkey key;
				Tvalue value;
				iar(key, value);

				this->set_value_recv(std::move(key), std::move(value), datas_recv);
			}
		}
		else if (this->flag_lock_set_value==Comm_Tools::Lock_Type::Lock_item)
		{
			for (size_t i=0; i<size_item; ++i)
			{
				Tkey key;
				Tvalue value;
				iar(key, value);

				while (lock_set_value.test_and_set(std::memory_order_seq_cst)) std::this_thread::yield();
				this->set_value_recv(std::move(key), std::move(value), datas_recv);
				lock_set_value.clear(std::memory_order_seq_cst);
			}
		}
		else if (this->flag_lock_set_value==Comm_Tools::Lock_Type::Lock_Process)
		{
			while (lock_set_value.test_and_set(std::memory_order_seq_cst)) std::this_thread::yield();
			for (size_t i=0; i<size_item; ++i)
			{
				Tkey key;
				Tvalue value;
				iar(key, value);

				this->set_value_recv(std::move(key), std::move(value), datas_recv);
			}
			lock_set_value.clear(std::memory_order_seq_cst);
		}
		else if (this->flag_lock_set_value==Comm_Tools::Lock_Type::Copy_merge)
		{
			Tdatas_recv datas_local = this->init_datas_local (rank_recv);
			for (size_t i=0; i<size_item; ++i)
			{
				Tkey key;
				Tvalue value;
				iar(key, value);

				this->set_value_recv (std::move(key), std::move(value), datas_local);
			}
			while (lock_set_value.test_and_set(std::memory_order_seq_cst)) std::this_thread::yield();
			this->add_datas (std::move(datas_local), datas_recv);
			lock_set_value.clear(std::memory_order_seq_cst);
		}
		else
		{
			throw std::invalid_argument(
				+" file "+std::string(__FILE__)
				+" line "+std::to_string(__LINE__)
				+" rank_mine "+std::to_string(this->rank_mine)
				+" rank_recv "+std::to_string(rank_recv));
		}
	} // end cereal::BinaryInputArchive
	buffer_recv.clear();
	buffer_recv.shrink_to_fit();
	state_recv = State_Recv::finish_iar;
}


template<typename Tkey, typename Tvalue, typename Tdatas_isend, typename Tdatas_recv>
bool Comm_Trans<Tkey,Tvalue,Tdatas_isend,Tdatas_recv>::check_finish(
	const std::vector<std::atomic<State_Send>> &states_send,
	const std::vector<std::atomic<State_Recv>> &states_recv) const
{
	for(int rank_isend_tmp=this->comm_size; rank_isend_tmp>0; --rank_isend_tmp)
	{
		const int rank_isend = (rank_isend_tmp + this->rank_mine) % this->comm_size;
		if(states_send[rank_isend] != State_Send::finish_isend)
			return false;
	}
	for(int rank_recv_tmp=0; rank_recv_tmp<this->comm_size; ++rank_recv_tmp)
	{
		const int rank_recv = (rank_recv_tmp + this->rank_mine) % this->comm_size;
		if(states_recv[rank_recv] != State_Recv::finish_iar)
			return false;
	}
	return true;
}

}

#undef MPI_CHECK

/*
get_send_keys()
{

	if(unique)
	{
		for(irank in all)
			send(irank_send, atom_pairs_remove);
	}
}
*/