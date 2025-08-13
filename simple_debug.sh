#!/bin/bash
cd /home/furina/powerline_extraction/point_cloud_history01/point_cloud_new_1

echo "=== 启动 GDB 调试 ==="
echo "程序运行并崩溃后会自动显示错误信息"
echo "在 (gdb) 提示符下输入 'run' 开始运行"
echo "崩溃后输入 'bt' 查看调用栈"

gdb --args ./devel/lib/powerline_extractor/powerline_extractor_node __name:=powerline_extractor
