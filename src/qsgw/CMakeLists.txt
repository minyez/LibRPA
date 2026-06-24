# All QSGW self-consistency modules — single registration point.
# src/CMakeLists.txt pulls this in via add_subdirectory(qsgw).
# Do NOT also list these files in src/CMakeLists.txt (no double-registration).
#
# Task A (infra):    qsgw_state / mixing / fermi_energy_occupation
# Task B (coremath): correlation_potential / hamiltonian_qsgw
# Task C (hartree):  hartree
target_sources(rpa_lib PRIVATE
    qsgw_state.cpp
    mixing.cpp
    fermi_energy_occupation.cpp
    correlation_potential.cpp
    hamiltonian_qsgw.cpp
    hartree.cpp
)
