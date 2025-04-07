"""
颜色检测模块
使用OpenCV实现基于HSV颜色空间的颜色检测
"""
import os
import sys
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # 禁用Media Foundation
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"      # 启用基本图像支持
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 颜色HSV范围定义（H: 0-180, S: 0-255, V: 0-255）
COLOR_RANGES = {
    "red":    [[[0, 150, 100], [10, 255, 255]], [[160, 150, 100], [179, 255, 255]]],
    "green":  [[[35, 50, 50], [85, 255, 255]]],
    "blue":   [[[90, 50, 50], [130, 255, 255]]],
    "yellow": [[[20, 100, 100], [35, 255, 255]]],
    "orange": [[[10, 100, 100], [20, 255, 255]]],
    "purple": [[[130, 50, 50], [160, 255, 255]]]
}

def detect_colors(img_path: str) -> None:
    """检测图片中的颜色区域并标注
    
    Args:
        img_path (str): 输入图片路径
        
    Raises:
        FileNotFoundError: 当图片文件不存在时
        ValueError: 当图片无法读取时
    """
    if getattr(sys, 'frozen', False):
        os.environ['TCL_LIBRARY'] = os.path.join(sys._MEIPASS, 'tcl', 'tcl8.6')
        os.environ['TK_LIBRARY'] = os.path.join(sys._MEIPASS, 'tk', 'tk8.6')
    try:
        # 读取图片并检查是否成功
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图片: {img_path}")
            
        logger.info(f"正在处理图片: {img_path}")
        
        # 预处理
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (11, 11), 0)

        # 遍历所有颜色
        for color_name, ranges in COLOR_RANGES.items():
            # 合并多个颜色区间
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

            # 形态学操作
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # 查找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 标注颜色区域
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:  # 过滤小面积区域
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, color_name, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 调整图像大小以适应窗口
        screen_res = 1280, 720  # 设置窗口大小
        scale_width = screen_res[0] / img.shape[1]
        scale_height = screen_res[1] / img.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(img.shape[1] * scale)
        window_height = int(img.shape[0] * scale)
        resized_img = cv2.resize(img, (window_width, window_height))

        # 显示结果
        #cv2.imshow("Color Detection", resized_img)
        #cv2.waitKey(0)

        # 转换为RGB格式
        resized_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # 创建新窗口显示结果
        result_window = tk.Toplevel()
        img_pil = Image.fromarray(resized_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        label = tk.Label(result_window, image=img_tk)
        label.image = img_tk  # 保持引用
        label.pack()

    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        raise
    except Exception as e:
        logger.error(f"处理图片时发生错误: {e}")
        raise
    finally:
        # 确保释放资源
        # cv2.destroyAllWindows()
        pass

def select_image():
    """打开文件对话框选择图片并进行颜色检测"""
    try:
        # 获取正确的初始目录
        initial_dir = os.path.expanduser("~")  # 从用户目录开始
        img_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            filetypes=[("图片文件", "*.jpg *.jpeg *.png")]
        )
        if img_path:
            logging.info(f"选择的文件路径: {img_path}")
            if not os.path.exists(img_path):
                logging.error("文件路径不存在")
                tk.messagebox.showerror("错误", "文件不存在")
                return
            
            detect_colors(img_path)
    except Exception as e:
        logging.exception("选择文件时发生异常:")
        tk.messagebox.showerror("系统错误", str(e))


if __name__ == "__main__":
    # 创建Tkinter主窗口
    root = tk.Tk()
    root.title("颜色检测系统")

    # 创建选择图片按钮
    btn_select = tk.Button(root, text="选择图片", command=select_image)
    btn_select.pack(pady=20)

    # 运行Tkinter主循环
    root.mainloop()