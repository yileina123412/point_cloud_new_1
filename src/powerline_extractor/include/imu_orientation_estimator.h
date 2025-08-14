#ifndef IMU_ORIENTATION_ESTIMATOR_H
#define IMU_ORIENTATION_ESTIMATOR_H

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <std_msgs/Bool.h>

#include <Eigen/Dense>
#include <memory>
#include <deque>
#include <string>

class IMUOrientationEstimator {
public:
    IMUOrientationEstimator(ros::NodeHandle& nh, ros::NodeHandle& private_nh);
    ~IMUOrientationEstimator();
    
    // 获取当前的姿态变换矩阵（用于点云校正）
    Eigen::Matrix4f getHorizontalTransformMatrix() const;
    
    // 获取当前的俯仰角和滚转角（弧度）
    double getCurrentPitch() const { return current_pitch_; }
    double getCurrentRoll() const { return current_roll_; }
    
    // 检查IMU数据是否有效
    bool isIMUDataValid() const { return imu_data_valid_; }
    
    // 获取当前重力向量在各轴的分量
    Eigen::Vector3f getCurrentGravityVector() const { return current_gravity_; }
    
    // 强制更新一次姿态估计（可选的手动触发）
    void forceUpdate();

    // 在public部分添加
    bool isDeviceStable() const { return is_device_stable_; }
    bool hasRecentMovement() const { return has_recent_movement_; }



private:
    // 回调函数
    void imuCallback(const sensor_msgs::Imu::ConstPtr& msg);
    
    // 参数加载
    void loadParameters();
    
    // 初始化订阅器
    void initializeSubscribers();
    
    // 坐标系转换
    bool transformAcceleration(const geometry_msgs::Vector3& input_accel,
                              const std_msgs::Header& header,
                              geometry_msgs::Vector3& output_accel);
    
    // 根据重力加速度计算姿态角
    void calculateOrientationFromGravity(const Eigen::Vector3f& gravity_vector);
    
    // 滤波函数
    void updateMovingAverage(const Eigen::Vector3f& new_accel);
    
    // 计算变换矩阵
    void updateTransformMatrix();
    
    // 验证IMU数据
    bool validateIMUData(const sensor_msgs::Imu::ConstPtr& msg);
    
    // 发布调试信息
    void publishDebugInfo(const std_msgs::Header& header);

    // 监察设备是否稳定
    void checkDeviceStability(const Eigen::Vector3f& current_accel, const ros::Time& current_time);


private:
    // ROS相关
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    ros::Subscriber imu_sub_;
    
    // TF变换
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    
    // 参数
    std::string imu_topic_;
    std::string imu_frame_;
    std::string target_frame_;
    
    // 滤波参数
    int filter_window_size_;
    double gravity_magnitude_;
    double accel_threshold_;
    double angle_threshold_;
    
    // 状态变量
    bool imu_data_valid_;
    double current_pitch_;  // 俯仰角（绕y轴旋转）
    double current_roll_;   // 滚转角（绕x轴旋转）
    Eigen::Vector3f current_gravity_;
    Eigen::Matrix4f horizontal_transform_matrix_;
    
    // 滤波用的历史数据
    std::deque<Eigen::Vector3f> accel_history_;
    Eigen::Vector3f filtered_accel_;
    
    // 时间戳管理
    ros::Time last_update_time_;
    double update_frequency_;
    
    // 调试发布器（可选）
    ros::Publisher gravity_vector_pub_;
    ros::Publisher orientation_pub_;

    // 稳定性检测相关
    bool is_device_stable_;
    bool has_recent_movement_;
    double stability_threshold_accel_;     // 加速度变化阈值
    double stability_threshold_angle_;     // 角度变化率阈值  
    double stability_required_time_;       // 要求的稳定时间
    ros::Time last_stable_time_;          // 上次稳定的时间
    ros::Time last_movement_time_;        // 上次检测到移动的时间

    // 历史数据用于变化率计算
    double prev_pitch_;
    double prev_roll_;
    Eigen::Vector3f prev_accel_;
    ros::Time prev_time_;

};

#endif // IMU_ORIENTATION_ESTIMATOR_H