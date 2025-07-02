#include "obstacle_analyzer.h"

#include <ros/ros.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/pca.h>
#include <algorithm>
#include <set>
#include <cmath>
#include <sstream>

// 读取参数
ObstacleAnalyzer::ObstacleAnalyzer(ros::NodeHandle& nh) {
    nh_ = nh; 
    nh.param("obstacle_analyze/cluster_tolerance", cluster_tolerance_, 0.2);
    nh.param("obstacle_analyze/cluster_min_size", cluster_min_size_, 30);
    nh.param("obstacle_analyze/cluster_max_size", cluster_max_size_, 100000);
    nh.param("obstacle_analyze/distance_search_step", distance_search_step_, 0.05); // 非必须

    ROS_INFO(
        "[ObstacleAnalyzer params] cluster_tolerance=%.2f min_size=%d max_size=%d search_step=%.2f",
        cluster_tolerance_, cluster_min_size_, cluster_max_size_, distance_search_step_);
}

void ObstacleAnalyzer::analyzeObstacles(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_cloud,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
    std::vector<OrientedBoundingBox>& obb_results)
{
    // 1. 障碍物分割
    pcl::PointCloud<pcl::PointXYZI>::Ptr obstacles_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    segmentObstacles(raw_cloud, powerline_cloud, obstacles_cloud);

    // 2. 欧式聚类
    std::vector<pcl::PointIndices> cluster_indices;
    euclideanClustering(obstacles_cloud, cluster_indices);

    // 3. 遍历每个障碍物
    obb_results.clear();
    for(const auto& indices : cluster_indices) {
        if(indices.indices.empty()) continue;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*obstacles_cloud, indices, *cluster_cloud);

        // 4. OBB计算
        OrientedBoundingBox obb;
        computeOrientedBoundingBox(cluster_cloud, obb);

        // 5. 最短距离
        obb.min_distance_to_powerline = computeMinDistance(cluster_cloud, powerline_cloud);

        obb_results.push_back(obb);
    }
    
}

// 1. 分割（用KDTree查找或hash唯一key查找更快，这里用暴力剔除）
void ObstacleAnalyzer::segmentObstacles(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& raw_cloud,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& obstacles_cloud)
{
    // 简便方案1: 用set做唯一判断（点云数量不大时）
    std::set<std::tuple<float,float,float>> powerline_pt_set;
    for(const auto& pt: powerline_cloud->points) {
        powerline_pt_set.insert( std::make_tuple(pt.x,pt.y,pt.z) );
    }
    obstacles_cloud->clear();
    for(const auto& pt: raw_cloud->points) {
        if(powerline_pt_set.count(std::make_tuple(pt.x,pt.y,pt.z))==0) {
            obstacles_cloud->push_back(pt);
        }
    }
    // 实际工程更可用KDTree,做欧拉距离判定
}

// 2. 欧式聚类
void ObstacleAnalyzer::euclideanClustering(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& obstacles_cloud,
    std::vector<pcl::PointIndices>& cluster_indices)
{
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(obstacles_cloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(cluster_min_size_);
    ec.setMaxClusterSize(cluster_max_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles_cloud);
    ec.extract(cluster_indices);
}

// 3. 姿态对齐OBB计算
void ObstacleAnalyzer::computeOrientedBoundingBox(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
    OrientedBoundingBox& obb)
{
    // 使用PCA (主方向)
    pcl::PCA<pcl::PointXYZI> pca;
    pca.setInputCloud(cluster_cloud);
    Eigen::Vector4f mean = pca.getMean();
    Eigen::Matrix3f eigenvectors = pca.getEigenVectors();

    // 坐标变换到主轴系
    std::vector<Eigen::Vector3f> transformed_pts;
    Eigen::Matrix4f tf = Eigen::Matrix4f::Identity();
    tf.block<3,3>(0,0) = eigenvectors.transpose();
    tf.block<3,1>(0,3) = -eigenvectors.transpose() * mean.head<3>();

    float minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9,minz=1e9,maxz=-1e9;
    for(const auto& pt: cluster_cloud->points) {
        Eigen::Vector4f ptv(pt.x, pt.y, pt.z, 1.0);
        Eigen::Vector4f local = tf * ptv;
        minx = std::min(minx, local[0]);
        maxx = std::max(maxx, local[0]);
        miny = std::min(miny, local[1]);
        maxy = std::max(maxy, local[1]);
        minz = std::min(minz, local[2]);
        maxz = std::max(maxz, local[2]);
    }

    // 得到OBB尺寸和中心
    Eigen::Vector3f mean_local( (minx+maxx)/2, (miny+maxy)/2, (minz+maxz)/2 );
    Eigen::Vector3f size( maxx-minx, maxy-miny, maxz-minz );

    // 坐标系变回世界系
    Eigen::Vector3f position = eigenvectors * mean_local + mean.head<3>();

    obb.position = position;
    obb.size = size;
    obb.rotation = eigenvectors;
}

// 4. 最近距离
float ObstacleAnalyzer::computeMinDistance(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_cloud,
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud)
{
    // 暴力法，也可以构建KDTree加速
    float min_dist = std::numeric_limits<float>::max();
    for(const auto& pt: cluster_cloud->points)
    {
        for(const auto& ptl: powerline_cloud->points)
        {
            float d = std::sqrt( pow(pt.x-ptl.x,2) + pow(pt.y-ptl.y,2) + pow(pt.z-ptl.z,2) );
            if(d < min_dist) min_dist = d;
        }
    }
    return min_dist;
}



void ObstacleAnalyzer::publishObbMarkers(
    const std::vector<OrientedBoundingBox>& obb_vec, 
    ros::Publisher& marker_pub,
    const std::string& frame_id
) {
    visualization_msgs::MarkerArray marker_array;

    for (size_t i = 0; i < obb_vec.size(); ++i) {
        const auto& obb = obb_vec[i];
        visualization_msgs::Marker box_marker;
        box_marker.header.frame_id = frame_id; // "map"或者你的点云坐标系
        box_marker.header.stamp = ros::Time::now();
        box_marker.ns = "obstacle_obb";
        box_marker.id = i;
        box_marker.type = visualization_msgs::Marker::CUBE;
        box_marker.action = visualization_msgs::Marker::ADD;

        // 位置
        box_marker.pose.position.x = obb.position.x();
        box_marker.pose.position.y = obb.position.y();
        box_marker.pose.position.z = obb.position.z();

        // 姿态
        Eigen::Matrix3f rot = obb.rotation;
        Eigen::Quaternionf q(rot);
        box_marker.pose.orientation.x = q.x();
        box_marker.pose.orientation.y = q.y();
        box_marker.pose.orientation.z = q.z();
        box_marker.pose.orientation.w = q.w();

        // 尺寸
        box_marker.scale.x = obb.size.x();
        box_marker.scale.y = obb.size.y();
        box_marker.scale.z = obb.size.z();

        // 颜色和透明度
        box_marker.color.r = 0.2;
        box_marker.color.g = 0.8;
        box_marker.color.b = 0.0;
        box_marker.color.a = 0.35;

        box_marker.lifetime = ros::Duration(0); // 可适当延长，单位秒

        marker_array.markers.push_back(box_marker);

        // 距离(文本)Marker
        visualization_msgs::Marker text_marker = box_marker;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.ns = "obstacle_distance";  // <--- 文字专用命名空间
        text_marker.id = 1000 + i; // id要和box_marker不同
        text_marker.pose.position.z += 0.5 * obb.size.z() + 0.1;
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << obb.min_distance_to_powerline << " m";
        text_marker.text = oss.str();
        text_marker.scale.x = text_marker.scale.y = 0.1;
        text_marker.scale.z = 0.2;        // 字高0.2米
        text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        marker_array.markers.push_back(text_marker);
    }

    for (size_t j = obb_vec.size(); j < g_last_num_obstacles; ++j) {
        // 删除多余的BOX
        visualization_msgs::Marker del_marker;
        del_marker.header.frame_id = frame_id;
        del_marker.header.stamp = ros::Time::now();
        del_marker.ns = "obstacle_obb";
        del_marker.id = j;
        del_marker.action = visualization_msgs::Marker::DELETE;
        marker_array.markers.push_back(del_marker);
    
        // 删除多余的文字
        visualization_msgs::Marker del_text;
        del_text.header = del_marker.header;
        del_text.header.frame_id = frame_id;
        del_text.ns = "obstacle_distance";
        del_text.id = 1000+j;
        del_text.action = visualization_msgs::Marker::DELETE;
        marker_array.markers.push_back(del_text);
    }

    marker_pub.publish(marker_array);
    g_last_num_obstacles = obb_vec.size(); // 记住本次，供下次比对
}



void ObstacleAnalyzer::publishPowerlineDistanceMarkers(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& powerline_cloud,
    ros::Publisher& marker_pub,
    const std::string& frame_id)
{
    visualization_msgs::MarkerArray marker_array;
    visualization_msgs::Marker points_marker;
    points_marker.header.frame_id = frame_id;
    points_marker.header.stamp = ros::Time::now();
    points_marker.ns = "powerline_dist";
    points_marker.id = 0;
    points_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    points_marker.scale.x = 0.07;  // 半径0.07米的球
    points_marker.scale.y = 0.07;
    points_marker.scale.z = 0.07;
    points_marker.action = visualization_msgs::Marker::ADD;
    points_marker.lifetime = ros::Duration(0); // 永久

    for(size_t i=0; i<powerline_cloud->points.size(); ++i) {
        const auto& pt = powerline_cloud->points[i];
        geometry_msgs::Point gm_pt;
        gm_pt.x = pt.x;
        gm_pt.y = pt.y;
        gm_pt.z = pt.z;
        points_marker.points.push_back(gm_pt);

        double dist = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);

        std_msgs::ColorRGBA color;
        // 彩色分段，可以自定义
        if(dist < 10.0) { // 0-10m 红色
            color.r = 1; color.g = 0; color.b = 0; color.a = 1.0;
        } else if(dist < 20.0) { // 10-20m 黄色
            color.r = 1; color.g = 1; color.b = 0; color.a = 1.0;
        } else if(dist < 30.0) {// 20-30m 绿
            color.r = 0; color.g = 1; color.b = 0; color.a = 1.0;
        } else {                // 30+m 蓝
            color.r = 0; color.g = 0; color.b = 1; color.a = 1.0;
        }
        points_marker.colors.push_back(color);

        // 可选：添加文本显示每个距离
        // 如果只显示较少点数量，建议加文本marker，否则太多太花了
        if(i % 50 == 0) { // 每50个点才加一个文字
            visualization_msgs::Marker text_marker;
            text_marker.header = points_marker.header;
            text_marker.ns = "powerline_dist_text";
            text_marker.id = 10000 + i;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::Marker::ADD;
            text_marker.pose.position = gm_pt;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(1) << dist << "m";
            text_marker.text = oss.str();
            text_marker.scale.z = 0.3;
            text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0;
            text_marker.color.a = 1.0;
            text_marker.lifetime = ros::Duration(0);
            marker_array.markers.push_back(text_marker);
        }
    }
    marker_array.markers.push_back(points_marker);

    marker_pub.publish(marker_array);
}