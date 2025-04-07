一、系统概述
1.1 功能特性
•	🎨 支持6种基础颜色识别（红、绿、蓝、黄、橙、紫）
•	📁 图形化文件选择界面（支持JPG/JPEG/PNG格式）
•	🖼️ 自适应窗口显示（自动缩放保持宽高比）
•	✅ 智能噪声过滤（形态学处理+面积阈值）
•	📝 完善的日志记录和错误处理
1.2 技术栈
模块	技术实现
核心算法	OpenCV 4.x
GUI界面	Tkinter
类型检查	Python Typing
日志系统	logging模块
图像处理	NumPy数组操作
二、核心算法说明
1．颜色检测流程
 ![image](https://github.com/user-attachments/assets/3d6f8f9b-7308-467b-80b5-198aec123c37)

2．关键参数配置
# 在COLOR_RANGES字典中调整HSV范围
"red": [
    [[0, 150, 100], [10, 255, 255]],   # 红色低区间
    [[160, 150, 100], [179, 255, 255]] # 红色高区间
]

# 形态学操作参数
cv2.erode(mask, None, iterations=2)   # 腐蚀迭代次数
cv2.dilate(mask, None, iterations=2)  # 膨胀迭代次数

# 面积过滤阈值
if cv2.contourArea(cnt) > 500:  # 最小有效区域(像素)

已知问题：
此问题是由于Python打包为EXE程序出现的缺少依赖，不影响Demo演示。点击确定即可。
 ![image](https://github.com/user-attachments/assets/17190ed7-1582-4bd9-a6e3-bbc79a1fb1fb)




