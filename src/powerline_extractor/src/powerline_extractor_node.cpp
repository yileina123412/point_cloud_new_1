#include "powerline_extractor.h"

int main(int argc, char** argv) {

    setlocale(LC_ALL, "");
    ros::init(argc, argv, "powerline_extractor_node");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");
    
    ROS_INFO("Starting Powerline Extractor Node...");
    
    try {
        PowerlineExtractor extractor(nh, private_nh);
        
        // 使用多线程处理回调
        ros::AsyncSpinner spinner(2);
        spinner.start();
        
        ROS_INFO("Powerline Extractor Node is ready and waiting for point cloud data...");
        
        ros::waitForShutdown();
        
        ROS_INFO("Shutting down Powerline Extractor Node");
    }
    catch (const std::exception& e) {
        ROS_ERROR("Exception in powerline extractor node: %s", e.what());
        return -1;
    }
    
    return 0;
}
