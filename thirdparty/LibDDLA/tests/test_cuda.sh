#!/bin/bash
#SBATCH -p v100g32
##SBATCH --nodelist gpu005
#SBATCH -J test
##SBATCH -A xgren
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --output=../../log_test
#SBATCH --error=../../err_test


module load gcc/11.3.0

module load openmpi/4.1.8-cuda
module load cmake/3.25.3

source /data/home/renxg/app/nvhpc/setup_nvhpc

cd ..
LibDDLA_PATH="${PWD}_install"
cd tests
export CPATH=$LibDDLA_PATH/include:$CPATH
export LIBRARY_PATH=$LibDDLA_PATH/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$LibDDLA_PATH/lib:$LD_LIBRARY_PATH


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

echo "任务运行节点列表: ${SLURM_NODELIST}"


echo Begin Time: `date`
### * * * Running the tasks * * * ###

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
    mpicxx -g -O2 -lcudart -lddla -fopenmp -lcublas -lcusolver -lcurand -lcal -lcusolverMp ${FILENAME}.cpp -o ../../${FILENAME} -std=c++17 -DDDLA_USE_CUDA -DDDLA_USE_CCL

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
    export OMPI_MCA_btl_openib_allow_ib=1
    mpirun -n $np --mca btl_tcp_if_include ib0,ib1 ../../${FILENAME}

    echo "✅ Finished: ${FILENAME}"
    echo "" # 空一行，方便看日志

done

echo "================================================="
echo "All tests finished."

echo End Time: `date`
