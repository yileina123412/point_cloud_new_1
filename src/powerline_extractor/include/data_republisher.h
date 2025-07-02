#ifndef DATA_REPUBLISHER_H
#define DATA_REPUBLISHER_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

class DataRepublisher {
private:
    ros::NodeHandle nh_;                  // ROS节点句柄
    ros::Subscriber powerline_sub_;       // 电力线点云订阅者
    ros::Subscriber env_sub_;             // 环境点云订阅者
    ros::Subscriber markers_sub_;         // 障碍物标记订阅者
    ros::Publisher powerline_pub_;        // 电力线点云发布者
    ros::Publisher env_pub_;              // 环境点云发布者
    ros::Publisher markers_pub_;          // 障碍物标记发布者
    ros::Timer timer_;                    // 定时器，用于1 Hz发布
    sensor_msgs::PointCloud2ConstPtr last_powerline_cloud_;  // 存储最新电力线点云
    sensor_msgs::PointCloud2ConstPtr last_env_cloud_;        // 存储最新环境点云
    visualization_msgs::MarkerArrayConstPtr last_markers_;   // 存储最新障碍物标记

public:
    DataRepublisher();                    // 构造函数
    void powerlineCallback(const sensor_msgs::PointCloud2ConstPtr& msg);  // 电力线回调
    void envCallback(const sensor_msgs::PointCloud2ConstPtr& msg);        // 环境回调
    void markersCallback(const visualization_msgs::MarkerArrayConstPtr& msg);  // 标记回调
    void timerCallback(const ros::TimerEvent& event);                     // 定时器回调
};

#endif // DATA_REPUBLISHER_H
