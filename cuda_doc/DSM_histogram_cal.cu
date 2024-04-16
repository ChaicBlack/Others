/* 当直方图太大而不能放入单个block的共享内存的时候，
   就可以考虑用DSM, DSM支持cluster内部block之间互相
   访问共享内存，这样就可以将直方图分段放入多个块中
   的共享内存*/
#include <cooperative_groups.h>

__global__ void clusterHist_kernel(int *bins, const int nbins,
                                   const int bins_per_block,
                                   const int *__restrict__ input,
                                   size_t array_size) {
  // 当前块的共享内存
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  // threadIdx across the whole grid
  int tid = cg::this_grid().thread_rank();

  // cluster initialization, size and local bin offsets.
  cg::cluster_group cluster = cg::this_cluster();
  int               cluster_size = cluster.dim_blocks().x;
  unsigned int      clusterBlockRank = cluster.block_rank();

  // 每个线程均匀初始化block共享内存中的几个数据
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x) {
    smem[i] = 0;
  }

  // 确保在cluster中有操作之前，所有block都启动了，确保能被其他block
  // 访问共享内存
  cluster.sync();
  // blockDim.x * gridDim.x = 网格中的线程总数
  // array_size是input数组的大小, 输入数组很可能比线程总数大很多
  // 每个线程都只能读取一个输入数组中的值
  for(int i = tid; i < array_size; i += blockDim.x * gridDim.x){
    int ldata = input[i];

    // 对读取的数据进行预处理
    int binid = ldata;
    if(ldata < 0)
      binid = 0;
    else if(ldata > nbins)
      binid = nbins - 1;

    // 通过binid找到blockid和offset
    int dst_block_rank = binid / bins_per_block;
    int dst_offset     = binid % bins_per_block;

    // 从当前clock的smem建立到dst_smem的映射，不用访问全局内存
    // 所以需要传入smem当作基准点, 找到需要访问的block的共享内存地址
    int* dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    // 对bin进行原子更新，每次加1
    atomicAdd(dst_smem + dst_offset, 1);
  }
  // 这个可以保证当cluster内部还有block在操作的时候，已经完成任务
  // 的block不会退出
  cluster.sync();

  // lbins是当前block负责计算直方图的起始位置
  int* lbins = bins + cluster.block_rank() * bins_per_block();
  for(int i = threadIdx.x; i < bins_per_block; i += blockDim.x){
    atomicAdd(&lbins[i], smem[i]);
  }
}
