#ifndef TASK_scRPA_H
#define TASK_scRPA_H
#include <map>
#include <vector>

#include "complexmatrix.h"
#include "vector3_order.h"




// 声明 scRPA 计算任务的函数
void task_scRPA(std::map<Vector3_Order<double>, ComplexMatrix>& sinvS);


#endif // TASK_scRPA_H
