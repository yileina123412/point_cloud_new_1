#include "data_republisher.h"

DataRepublisher::DataRepublisher() : nh_("~") {
    // 获取参数
    std::string powerline_input_topic;
    nh_.param<std::string>("powerline_input_topic", powerline_input_topic, "/powerline_pointcloud");
    std::string powerline_output_topic;
    nh_.param<std::string>("powerline_output_topic", powerline_output_topic, "/republished_powerline");

    std::string env_input_topic;
    nh_.param<std::string>("env_input_topic", env_input_topic, "/env_pointcloud");
    std::string env_output_topic;
    nh_.param<std::string>("env_output_topic", env_output_topic, "/republished_env");

    std::string markers_input_topic;
    nh_.param<std::string>("markers_input_topic", markers_input_topic, "/obb_marker");
    std::string markers_output_topic;
    nh_.param<std::string>("markers_output_topic", markers_output_topic, "/republished_markers");

    double republish_rate;
    nh_.param<double>("republish_rate", republish_rate, 1.0);
    ROS_INFO("powerline_input_topic is %s",powerline_input_topic.c_str());

    // 初始化订阅者
    powerline_sub_ = nh_.subscribe(powerline_input_topic, 1, &DataRepublisher::powerlineCallback, this);
    env_sub_ = nh_.subscribe(env_input_topic, 1, &DataRepublisher::envCallback, this);
    markers_sub_ = nh_.subscribe(markers_input_topic, 1, &DataRepublisher::markersCallback, this);

    // 初始化发布者
    powerline_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(powerline_output_topic, 1);
    env_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(env_output_topic, 1);
    markers_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(markers_output_topic, 10);

    // 初始化定时器
    timer_ = nh_.createTimer(ros::Duration(1.0 / republish_rate), &DataRepublisher::timerCallback, this);
}

void DataRepublisher::powerlineCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    last_powerline_cloud_ = msg;  // 更新最新电力线点云
}

void DataRepublisher::envCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    last_env_cloud_ = msg;        // 更新最新环境点云
}

void DataRepublisher::markersCallback(const visualization_msgs::MarkerArrayConstPtr& msg) {
    last_markers_ = msg;          // 更新最新障碍物标记
}

void DataRepublisher::timerCallback(const ros::TimerEvent& event) {
    // 如果收到过消息，则发布
    if (last_powerline_cloud_) {
        powerline_pub_.publish(*last_powerline_cloud_);
    }
    if (last_env_cloud_) {
        env_pub_.publish(*last_env_cloud_);
    }
    if (last_markers_) {
        markers_pub_.publish(*last_markers_);
    }
}
