#include <iostream>
using namespace std; 
// 定义算法类的父类
class Aigorithm{
public:
    virtual void Aigorithm_init() = 0;
    virtual void Aigorithm_interface() = 0;
    virtual void Aigorithm_finalize() = 0;
    virtual ~Aigorithm() = default;
};

// 算法A(继承算法类)   (yolov5s.engine)
class yolov5 : public Aigorithm{
public:
   void Aigorithm_init() override;    //重写算法初始化
   void Aigorithm_interface() override;//重写算法推理
   void Aigorithm_finalize() override;  //重写算法清理
};