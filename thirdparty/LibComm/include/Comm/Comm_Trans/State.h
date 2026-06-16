//=======================
// AUTHOR : Peize Lin
// DATE :   2026-05-04
//=======================

#pragma once

namespace Comm
{

enum class State_Send {unstart, begin_oar, finish_oar, begin_isend, finish_isend};
enum class State_Recv {unstart, begin_recv, finish_recv, begin_iar, finish_iar};

}