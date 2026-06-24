#include "qsgw_state.h"

namespace librpa_int {

void QsgwState::snapshot_wfc0(const MeanField &mf)
{
    // Deep copy of the [ispin][ispinor][ikpt] eigenvector map.
    wfc0 = mf.get_eigenvectors();
    has_wfc0 = true;
}

void QsgwState::snapshot_wg0(const MeanField &mf)
{
    // Deep copy of the [ispin](ikpt, ib) occupation-weight vector.
    wg0 = mf.get_weight();
    has_wg0 = true;
}

void QsgwState::snapshot_velocity0(
    const std::vector<std::vector<std::vector<ComplexMatrix>>> &velocity)
{
    velocity0 = velocity;
    has_velocity0 = true;
}

void QsgwState::snapshot(
    const MeanField &mf,
    const std::vector<std::vector<std::vector<ComplexMatrix>>> &velocity)
{
    snapshot_wfc0(mf);
    snapshot_wg0(mf);
    snapshot_velocity0(velocity);
}

const ComplexMatrix *QsgwState::find_wfc0(int ispin, int ispinor, int ikpt) const noexcept
{
    auto it_s = wfc0.find(ispin);
    if (it_s == wfc0.end()) return nullptr;
    auto it_so = it_s->second.find(ispinor);
    if (it_so == it_s->second.end()) return nullptr;
    auto it_k = it_so->second.find(ikpt);
    if (it_k == it_so->second.end()) return nullptr;
    return &it_k->second;
}

void QsgwState::clear()
{
    wfc0.clear();
    wg0.clear();
    velocity0.clear();
    has_wfc0 = false;
    has_wg0 = false;
    has_velocity0 = false;
}

} // namespace librpa_int
