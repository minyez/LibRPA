CXX ?= c++
MPICXX ?= mpicxx
CXXFLAGS ?= -std=c++17 -O2

.PHONY: all clean

SRCS = convert_legacy_Cs.cpp convert_legacy_coulomb_mat_mpi.cpp
OBJS = $(patsubst %.cpp,%.exe,$(SRCS))

all: $(OBJS)

convert_legacy_Cs.exe: convert_legacy_Cs.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

convert_legacy_coulomb_mat_mpi.exe: convert_legacy_coulomb_mat_mpi.cpp
	$(MPICXX) $(CXXFLAGS) $< -o $@

clean:
	$(RM) -r $(OBJS) *.dSYM
