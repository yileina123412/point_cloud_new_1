#include "imu_orientation_estimator.h"
#include <tf2/utils.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <std_msgs/Float64MultiArray.h>

IMUOrientationEstimator::IMUOrientationEstimator(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), 
      private_nh_(private_nh),
      imu_data_valid_(false),
      current_pitch_(0.0),
      current_roll_(0.0),
      current_gravity_(Eigen::Vector3f(0.0f, 0.0f, 9.8f)),
      filtered_accel_(Eigen::Vector3f::Zero()),
      horizontal_transform_matrix_(Eigen::Matrix4f::Identity()) {
    
    // 初始化TF
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    
    // 加载参数
    loadParameters();
    
    // 初始化订阅器
    initializeSubscribers();
    
    // 初始化调试发布器
    gravity_vector_pub_ = private_nh_.advertise<geometry_msgs::Vector3Stamped>("gravity_vector", 1);
    orientation_pub_ = private_nh_.advertise<std_msgs::Float64MultiArray>("orientation_angles", 1);
    
    ROS_INFO("IMU Orientation Estimator initialized successfully");
    ROS_INFO("Subscribing to IMU topic: %s", imu_topic_.c_str());
    ROS_INFO("IMU frame: %s, Target frame: %s", imu_frame_.c_str(), target_frame_.c_str());
}

IMUOrientationEstimator::~IMUOrientationEstimator() {
    ROS_INFO("IMU Orientation Estimator destructor called");
}

void IMUOrientationEstimator::loadParameters() {
    // IMU相关参数
    private_nh_.param<std::string>("imu_topic", imu_topic_, "/wit/imu");
    private_nh_.param<std::string>("imu_frame", imu_frame_, "base_link");
    private_nh_.param<std::string>("target_frame", target_frame_, "map");
    
    // 滤波参数
    private_nh_.param<int>("imu_filter_window_size", filter_window_size_, 10);
    private_nh_.param<double>("gravity_magnitude", gravity_magnitude_, 9.8);
    private_nh_.param<double>("imu_accel_threshold", accel_threshold_, 0.5);
    private_nh_.param<double>("imu_angle_threshold", angle_threshold_, 0.02);  // ~1度
    private_nh_.param<double>("imu_update_frequency", update_frequency_, 10.0);
    
    // 打印参数
    ROS_INFO("=== IMU Orientation Estimator Parameters ===");
    ROS_INFO("IMU topic: %s", imu_topic_.c_str());
    ROS_INFO("IMU frame: %s", imu_frame_.c_str());
    ROS_INFO("Target frame: %s", target_frame_.c_str());
    ROS_INFO("Filter window size: %d", filter_window_size_);
    ROS_INFO("Gravity magnitude: %.2f m/s²", gravity_magnitude_);
    ROS_INFO("Acceleration threshold: %.3f m/s²", accel_threshold_);
    ROS_INFO("Angle threshold: %.4f rad (%.2f°)", angle_threshold_, angle_threshold_ * 180.0 / M_PI);
    ROS_INFO("Update frequency: %.1f Hz", update_frequency_);
}

void IMUOrientationEstimator::initializeSubscribers() {
    imu_sub_ = nh_.subscribe(imu_topic_, 1, &IMUOrientationEstimator::imuCallback, this);
}

void IMUOrientationEstimator::imuCallback(const sensor_msgs::Imu::ConstPtr& msg) {
    // 频率控制
    ros::Time current_time = ros::Time::now();
    if (!last_update_time_.isZero() && 
        (current_time - last_update_time_).toSec() < (1.0 / update_frequency_)) {
        return;
    }
    last_update_time_ = current_time;
    
    // 验证IMU数据
    if (!validateIMUData(msg)) {
        ROS_WARN_THROTTLE(5.0, "Invalid IMU data received");
        return;
    }
    
    try {
        // 坐标系转换
        geometry_msgs::Vector3 transformed_accel;
        if (!transformAcceleration(msg->linear_acceleration, msg->header, transformed_accel)) {
            ROS_WARN_THROTTLE(5.0, "Failed to transform IMU acceleration data");
            return;
        }
        
        // 转换为Eigen向量
        Eigen::Vector3f accel_vector(
            transformed_accel.x,
            transformed_accel.y,
            transformed_accel.z
        );
        
        // 更新滑动平均滤波
        updateMovingAverage(accel_vector);
        
        // 根据滤波后的重力向量计算姿态
        calculateOrientationFromGravity(filtered_accel_);
        
        // 更新变换矩阵
        updateTransformMatrix();
        
        // 标记数据有效
        imu_data_valid_ = true;
        
        // 发布调试信息
        publishDebugInfo(msg->header);
        
    } catch (const std::exception& e) {
        ROS_ERROR("Error processing IMU data: %s", e.what());
        imu_data_valid_ = false;
    }
}

bool IMUOrientationEstimator::transformAcceleration(const geometry_msgs::Vector3& input_accel,
                                                   const std_msgs::Header& header,
                                                   geometry_msgs::Vector3& output_accel) {
    try {
        // 如果已经在目标坐标系中，直接返回
        if (header.frame_id == target_frame_) {
            output_accel = input_accel;
            return true;
        }
        
        // 查找变换
        geometry_msgs::TransformStamped transform_stamped;
        transform_stamped = tf_buffer_->lookupTransform(
            target_frame_, header.frame_id,
            header.stamp, ros::Duration(1.0));
        
        // 创建Vector3Stamped进行变换
        geometry_msgs::Vector3Stamped input_stamped, output_stamped;
        input_stamped.header = header;
        input_stamped.vector = input_accel;
        
        // 执行变换
        tf2::doTransform(input_stamped, output_stamped, transform_stamped);
        output_accel = output_stamped.vector;
        
        return true;
        
    } catch (tf2::TransformException& ex) {
        ROS_WARN_THROTTLE(5.0, "Could not transform IMU acceleration: %s", ex.what());
        return false;
    }
}

void IMUOrientationEstimator::updateMovingAverage(const Eigen::Vector3f& new_accel) {
    // 添加新数据
    accel_history_.push_back(new_accel);
    
    // 保持窗口大小
    if (accel_history_.size() > static_cast<size_t>(filter_window_size_)) {
        accel_history_.pop_front();
    }
    
    // 计算滑动平均
    filtered_accel_ = Eigen::Vector3f::Zero();
    for (const auto& accel : accel_history_) {
        filtered_accel_ += accel;
    }
    filtered_accel_ /= static_cast<float>(accel_history_.size());
}

void IMUOrientationEstimator::calculateOrientationFromGravity(const Eigen::Vector3f& gravity_vector) {
    // 更新当前重力向量
    current_gravity_ = gravity_vector;
    
    // 提取重力分量
    float gx = gravity_vector.x();
    float gy = gravity_vector.y();
    float gz = gravity_vector.z();
    
    // 计算重力向量的模长
    float gravity_norm = gravity_vector.norm();
    
    // 检查重力向量是否合理
    if (std::abs(gravity_norm - gravity_magnitude_) > accel_threshold_) {
        ROS_WARN_THROTTLE(10.0, "Gravity magnitude deviation: %.3f (expected: %.3f)", 
                         gravity_norm, gravity_magnitude_);
    }
    
    // 归一化重力向量
    if (gravity_norm < 0.1f) {
        ROS_WARN_THROTTLE(5.0, "Gravity vector too small: %.3f", gravity_norm);
        return;
    }
    
    Eigen::Vector3f normalized_gravity = gravity_vector / gravity_norm;
    gx = normalized_gravity.x();
    gy = normalized_gravity.y();
    gz = normalized_gravity.z();
    
    // 计算俯仰角（pitch）- 绕y轴旋转
    // 在静止状态下，俯仰角影响x和z轴的重力分量
    double new_pitch = std::atan2(gy, std::sqrt(gx * gx + gz * gz));
    
    // 计算滚转角（roll）- 绕x轴旋转
    // 在静止状态下，滚转角影响y和z轴的重力分量
    double new_roll = std::atan2(gx, gz);
    
    // 角度变化检测和平滑
    if (std::abs(new_pitch - current_pitch_) > angle_threshold_ ||
        std::abs(new_roll - current_roll_) > angle_threshold_) {
        
        current_pitch_ = new_pitch;
        current_roll_ = new_roll;
        
        ROS_DEBUG("Updated orientation - Pitch: %.2f°, Roll: %.2f°", 
                 current_pitch_ * 180.0 / M_PI, current_roll_ * 180.0 / M_PI);
    }
}

void IMUOrientationEstimator::updateTransformMatrix() {
    // 创建旋转矩阵以将激光雷达转换到水平姿态
    // 注意：这里的变换是为了将倾斜的点云校正到水平状态
    
    // 创建绕x轴（roll）和y轴（pitch）的旋转矩阵
    Eigen::AngleAxisf roll_rotation(-current_roll_, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitch_rotation(-current_pitch_, Eigen::Vector3f::UnitY());
    
    // 组合旋转矩阵
    Eigen::Matrix3f rotation_matrix = pitch_rotation.toRotationMatrix() * roll_rotation.toRotationMatrix();
    
    // 构建4x4变换矩阵
    horizontal_transform_matrix_ = Eigen::Matrix4f::Identity();
    horizontal_transform_matrix_.block<3,3>(0,0) = rotation_matrix;
}

Eigen::Matrix4f IMUOrientationEstimator::getHorizontalTransformMatrix() const {
    return horizontal_transform_matrix_;
}

bool IMUOrientationEstimator::validateIMUData(const sensor_msgs::Imu::ConstPtr& msg) {
    // 检查时间戳
    if (msg->header.stamp.isZero()) {
        return false;
    }
    
    // 检查加速度数据是否有效（不是NaN或无穷大）
    const auto& accel = msg->linear_acceleration;
    if (!std::isfinite(accel.x) || !std::isfinite(accel.y) || !std::isfinite(accel.z)) {
        return false;
    }
    
    // 检查加速度幅值是否在合理范围内
    double accel_magnitude = std::sqrt(accel.x * accel.x + accel.y * accel.y + accel.z * accel.z);
    if (accel_magnitude < 1.0 || accel_magnitude > 20.0) {  // 合理的重力加速度范围
        return false;
    }
    
    return true;
}

void IMUOrientationEstimator::publishDebugInfo(const std_msgs::Header& header) {
    // 发布重力向量
    if (gravity_vector_pub_.getNumSubscribers() > 0) {
        geometry_msgs::Vector3Stamped gravity_msg;
        gravity_msg.header = header;
        gravity_msg.header.frame_id = target_frame_;
        gravity_msg.vector.x = current_gravity_.x();
        gravity_msg.vector.y = current_gravity_.y();
        gravity_msg.vector.z = current_gravity_.z();
        gravity_vector_pub_.publish(gravity_msg);
    }
    
    // 发布姿态角
    if (orientation_pub_.getNumSubscribers() > 0) {
        std_msgs::Float64MultiArray orientation_msg;
        orientation_msg.data.resize(2);
        orientation_msg.data[0] = current_pitch_;  // 俯仰角
        orientation_msg.data[1] = current_roll_;   // 滚转角
        orientation_pub_.publish(orientation_msg);
    }
}

void IMUOrientationEstimator::forceUpdate() {
    if (!accel_history_.empty()) {
        calculateOrientationFromGravity(filtered_accel_);
        updateTransformMatrix();
    }
}