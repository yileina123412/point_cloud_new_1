#include "data_republisher.h"

int main(int argc, char** argv) {
    ros::init(argc, argv, "data_republisher");  // 初始化ROS节点
    DataRepublisher republisher;                // 创建DataRepublisher实例
    ros::spin();                                // 进入事件循环
    return 0;
}
