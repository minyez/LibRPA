#include <ddla/ddla_connector.h>
#include <chrono>
#include <fstream>

namespace ddla{

void random_generator(void* c_data, const int64_t& lengthOfData, const deviceDataType_t& compute_type)
{
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    derandGenerator_t gen;
    DERAND_CHECK(derandCreateGenerator(&gen, DERAND_RNG_PSEUDO_DEFAULT));
    DERAND_CHECK(derandSetPseudoRandomGeneratorSeed(gen, static_cast<unsigned long long>(seed)));
    if(compute_type == DEVICE_C_64F)
        DERAND_CHECK(derandGenerateUniformDouble(gen,(double*) c_data, lengthOfData*2));
    else if(compute_type == DEVICE_C_32F)
        DERAND_CHECK(derandGenerateUniform(gen, (float*)c_data, lengthOfData*2));
    else if(compute_type == DEVICE_R_64F)
        DERAND_CHECK(derandGenerateUniformDouble(gen,(double*) c_data, lengthOfData));
    else if(compute_type == DEVICE_R_32F)
        DERAND_CHECK(derandGenerateUniform(gen, (float*)c_data, lengthOfData));
    else {
        std::cerr << "Unsupported compute type!" << std::endl;
        return;
    }
    DERAND_CHECK(derandDestroyGenerator(gen));
    return;
}

void write_matrix(std::complex<double>* A, const int& m,const int& n, const char* filename)
{

    std::ofstream outfile;
    outfile.open(filename, std::ios::out | std::ios::trunc);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            outfile<<"(";
            if(std::abs(A[i+j*m].real())<1e-10)
                outfile<<"0";
            else
                outfile<<A[i+j*m].real();
            outfile<<",";
            if(std::abs(A[i+j*m].imag())<1e-10)
                outfile<<"0";
            else
                outfile<<A[i+j*m].imag();
            outfile<<") ";
        }
        outfile<<"\n";
    }
    outfile.close();
    return;
}


} // namespace ddla