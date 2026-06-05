CXX ?= c++
MPICXX ?= mpicxx
CXXFLAGS ?= -std=c++17 -O2

.PHONY: all clean

SRCS = convert_legacy_Cs_v1.cpp convert_legacy_coulomb_mat_v1_mpi.cpp
OBJS = $(patsubst %.cpp,%.exe,$(SRCS))

all: $(OBJS)

convert_legacy_Cs_v1.exe: convert_legacy_Cs_v1.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

convert_legacy_coulomb_mat_v1_mpi.exe: convert_legacy_coulomb_mat_v1_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

clean:
	$(RM) -r $(OBJS) *.dSYM
