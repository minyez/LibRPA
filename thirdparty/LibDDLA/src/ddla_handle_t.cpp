#include <ddla/ddla_handle_t.h>
#include <ddla/ddla_stream.h>

namespace ddla{
void ddla_init(DdlaStream*& ddla_handle){
    ddla_handle = new DdlaStream();
    return;
}

void ddla_set(DdlaStream* ddla_handle, const MPI_Comm& comm, const char& major){
    ddla_handle->init(comm, major);
    return;
}

void ddla_set(DdlaHandle_t ddla_handle, const MPI_Comm& comm, const int& nprows, const int& npcols, const char& major){
    ddla_handle->init(nprows, npcols, comm, major);
    return;
}

void ddla_destroy(DdlaHandle_t& ddla_handle){
    if(ddla_handle!=nullptr){
        ddla_handle->clean();
        delete ddla_handle;
        ddla_handle = nullptr;
    }
    return;
}
}