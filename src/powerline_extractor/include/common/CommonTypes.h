#ifndef COMMON_H
#define COMMON_H
#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


struct CoarseExtractorSeedPoint {
    Eigen::Vector3f center_point; // 种子段中心点
    pcl::PointCloud<pcl::PointXYZI>::Ptr points;  // 种子段点云
    double confidence;              // 置信度 [0,1]
    Eigen::Vector3f direction;      // 主方向向量  单位向量
    ros::Time timestamp;            // 时间戳
    int ID;
    double length; // 种子段延主方向的长度（单位：米）
    CoarseExtractorSeedPoint() : confidence(0.0),length(0), timestamp(ros::Time::now()) {}
};




#endif