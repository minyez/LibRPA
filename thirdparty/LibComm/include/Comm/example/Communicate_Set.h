// ===================
//  Author: Peize Lin
//  date: 2022.06.22
// ===================

#pragma once

#include <set>
#include <functional>

namespace Comm
{

namespace Communicate_Set
{
	template<typename Tkey>
	void traverse_keys(
		const std::set<Tkey> &keys,
		std::function<void(const Tkey&)> &func)
	{
		for(const Tkey &key : keys)
			func(key);
	}

}

}