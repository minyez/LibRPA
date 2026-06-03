#!/bin/bash
#SBATCH -p normal
##SBATCH --nodelist j14r3n06
#SBATCH -J test
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=dcu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --output=../log_hip
#SBATCH --error=../err_hip

ulimit -s unlimited
ulimit -c unlimited


unset CPATH

module purge

module load compiler/rocm/dtk/25.04.3
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64/:$LIBRARY_PATH
module load compiler/devtoolset/9.3.1
module load mpi/hpcx/2.13.1/gcc-9.3.1-wangxh

LibDDLA_PATH="${PWD}_install"
cd tests
export CPATH=$LibDDLA_PATH/include:$CPATH
export LIBRARY_PATH=$LibDDLA_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$LibDDLA_PATH/lib:$LD_LIBRARY_PATH

export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64/:$LIBRARY_PATH
export CPATH=$ROCM_PATH/include/rocrand:$CPATH
echo "========================="
echo 'LD_LIBRARY_PATH:' $LD_LIBRARY_PATH
echo "========================="
echo 'PATH:' $PATH
echo "========================="
echo 'CPATH:' $CPATH
echo "========================="
echo 'C_INCLUDE_PATH:' $C_INCLUDE_PATH
echo "========================="
echo 'LIBRARY_PATH:' $LIBRARY_PATH
echo "========================="
echo 'CPLUS_INCLUDE_PATH:' $CPLUS_INCLUDE_PATH
echo "========================="

export LANGUAGE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

echo "任务运行节点列表: ${SLURM_NODELIST}"

echo Begin Time: `date`
### * * * Running the tasks * * * ###

# FILENAME=test_pzgemm
files=(
    "test_sv_gemm"
    # "test_aware"
    # "test_pgeadd"
    # "test_potrf_solvermp"
    "test_potrf_potrs"
)

# 遍历数组中的每一个文件
for FILENAME in "${files[@]}"; do
    rm ../../${FILENAME}
    
    echo "================================================="
    echo "🚀 Processing: ${FILENAME}"

    # 2. 编译阶段 (注意源文件路径加了 ./)
    echo "⏳ Compiling..."
    mpicxx -gdwarf-4 -g -O2 -lamdhip64 -lgalaxyhip -lddla -fopenmp -lrccl -lhipblas -lhipsolver -lhiprand  ${FILENAME}.cpp -o ../../${FILENAME} -std=c++17 -DDDLA_USE_HIP -D__HIP_PLATFORM_AMD__ -DDDLA_USE_CCL
    # 检查编译是否成功
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to compile ${FILENAME}"
        continue # 如果编译失败，跳过本次循环，继续下一个
    fi

    # 3. 计算进程数 (沿用你原来的逻辑)
    np=$((SLURM_NTASKS_PER_NODE * SLURM_NNODES))
    echo "📊 NP: $np"

    # 4. 运行阶段
    echo "▶️ Running..."
    # 这里保留了你原来的 mpirun 逻辑

    mpirun -n $np ../../${FILENAME} --mca btl ^openib

    echo "✅ Finished: ${FILENAME}"
    echo "" # 空一行，方便看日志

done

echo "================================================="
echo "All tests finished."

echo End Time: `date`

