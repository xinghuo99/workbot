"""workbot - 基于 AI 视觉模型的 GUI 自动化机器人"""

import os
import base64
import cv2
import numpy as np
from openai import OpenAI
import json
import re
import time
import pyautogui
import pyperclip
import signal
import platform
from pydantic import BaseModel
from utils.screenshot import mark_coordinate_on_image, capture_screen_and_save, map_coordinates
from utils.role import WORKBOT_ROLE
from utils.log import setup_workbot_logger


class MathResponse(BaseModel):
    current_status: str
    solving_problem: str
    whether_completed: str
    element_info: str
    coordinates: list
    action: str
    type_information: str


class WorkBot:
    """GUI 自动化机器人类，封装截屏、AI 分析、鼠标键盘控制等功能"""

    # ==================== 默认配置 ====================
    DEFAULT_CONFIG = {
        "api_config": {
            "api_key": "",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model_name": "doubao-seed-1-6-vision-250815"
        },
        "ai_config": {
            "enable_thinking": False,
            "thinking_type": "disabled",
            "vl_high_resolution_images": True
        },
        "execution_config": {
            "max_visual_model_iterations": 80,
            "default_max_iterations": 15
        },
        "screenshot_config": {
            "optimize_for_speed": True,
            "max_png": 1280
        },
        "mouse_config": {
            "move_duration": 0.1,
            "failsafe": False,
        }
    }

    # ==================== 初始化 ====================

    def __init__(self, config_path="config.json",
                 base_dir="xiaohua/workbot",
                 input_file="data/input_message.json",
                 output_file="data/output_message.json"):
        """
        初始化 WorkBot 实例

        :param config_path: 配置文件路径
        :param base_dir: 基础存储目录（图片和日志的父目录）
        :param input_file: 进程间通信输入文件路径
        :param output_file: 进程间通信输出文件路径
        """
        # 路径配置
        self._base_dir = base_dir
        self._images_dir = os.path.join(base_dir, "images")
        self._logs_dir = os.path.join(base_dir, "logs")
        self._config_path = config_path
        self._input_file = input_file
        self._output_file = output_file

        # 截图相关路径
        self._input_path = os.path.join(self._images_dir, "screen.png")
        self._output_path = os.path.join(self._images_dir, "screen_label")

        # 确保目录存在
        os.makedirs(self._images_dir, exist_ok=True)
        os.makedirs(self._logs_dir, exist_ok=True)

        # 初始化日志
        self._setup_logger()

        # 加载配置
        self._load_config()

        # 运行时状态
        self._should_exit = False
        self._global_client = None
        self._coordinate_callback = None
        self._current_os = platform.system()

        # 设置 pyautogui 安全机制
        pyautogui.FAILSAFE = self._mouse_config["failsafe"]

        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)

        # 尝试导入日志窗口模块
        try:
            from log_window import get_log_window
            self._log_window = get_log_window()
            self._log_window_available = True
        except ImportError:
            self._log_window = None
            self._log_window_available = False

        self.logger.info("WorkBot 初始化完成")

    # ==================== 日志模块 ====================

    def _setup_logger(self):
        """初始化日志模块，日志文件存放在 xiaohua/workbot/logs/ 下"""
        self.logger = setup_workbot_logger(self._logs_dir)

    def log(self, *args, **kwargs):
        """统一的日志打印方法"""
        msg = " ".join(str(arg) for arg in args)
        self.logger.info(msg)

    # ==================== 配置加载 ====================

    def _load_config(self):
        """加载配置文件，若失败则使用默认配置"""
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.log(f"成功加载配置文件: {self._config_path}")
        except Exception as e:
            self.log(f"加载配置文件失败: {e}，使用默认配置")
            config = None

        if config:
            self._api_config = config.get("api_config", self.DEFAULT_CONFIG["api_config"])
            self._ai_config = config.get("ai_config", self.DEFAULT_CONFIG["ai_config"])
            self._execution_config = config.get("execution_config", self.DEFAULT_CONFIG["execution_config"])
            self._screenshot_config = config.get("screenshot_config", self.DEFAULT_CONFIG["screenshot_config"])
            self._mouse_config = config.get("mouse_config", self.DEFAULT_CONFIG["mouse_config"])
        else:
            self._api_config = self.DEFAULT_CONFIG["api_config"]
            self._ai_config = self.DEFAULT_CONFIG["ai_config"]
            self._execution_config = self.DEFAULT_CONFIG["execution_config"]
            self._screenshot_config = self.DEFAULT_CONFIG["screenshot_config"]
            self._mouse_config = self.DEFAULT_CONFIG["mouse_config"]

        # 提取模型相关属性
        self._compute_model_properties()

    def reload_config(self):
        """重新加载配置文件"""
        self._load_config()

    def _compute_model_properties(self):
        """从配置中提取模型相关属性，统一管理"""
        self._api_key = self._api_config.get("api_key", "")
        self._base_url = self._api_config.get("base_url", "")
        self._model_name = self._api_config.get("model_name", "")
        self._thinking_type = self._ai_config.get("thinking_type", "disabled")
        self._is_volcano = (self._base_url == "https://ark.cn-beijing.volces.com/api/v3")

    # ==================== 模型客户端 ====================

    def _get_client(self):
        """获取或创建 OpenAI 客户端（懒初始化）"""
        if self._global_client is None:
            if not self._api_key:
                self.log("未设置 API Key，无法创建客户端")
                return None
            self._global_client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._global_client

    # ==================== 属性访问 ====================

    @property
    def images_dir(self):
        return self._images_dir

    @property
    def logs_dir(self):
        return self._logs_dir

    @property
    def input_path(self):
        return self._input_path

    @property
    def output_path(self):
        return self._output_path

    # ==================== 回调设置 ====================

    def set_coordinate_callback(self, callback):
        """设置坐标回调函数，用于通知主窗口 AI 输出的坐标"""
        self._coordinate_callback = callback

    # ==================== 停止控制 ====================

    def stop(self):
        """停止客户端连接"""
        self._should_exit = True
        self.log("已设置退出标志，等待API调用完成...")

        if self._global_client is not None:
            try:
                self.log("正在关闭与远程AI服务器的连接...")
                self._global_client.close()
                self.log("已关闭客户端连接")
            except Exception as e:
                self.log(f"关闭客户端连接时出错: {e}")
            finally:
                self._global_client = None

    def _signal_handler(self, sig, frame):
        """信号处理函数"""
        self.log("\n\n收到中断信号 (Ctrl+C)，正在停止执行...")
        self._should_exit = True
        self.stop()

    # ==================== 进程间通信 ====================

    def fresh_display_info_vl(self, content):
        """将信息写入输出文件，用于进程间通信"""
        try:
            with open(self._output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {'content': ''}

        old_content = data.get('content', '')
        new_content = old_content + "\n" + content
        response_data = {
            'request_id': str(time.time()),
            'content': new_content,
            'timestamp': time.time()
        }

        with open(self._output_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False)

    # ==================== 图片处理 ====================

    def read_local_image(self, image_path=None):
        """读取本地图片并转换为 base64 编码"""
        if image_path is None:
            image_path = self._input_path

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise Exception(f"无法读取图片: {image_path}")

            height, width, channels = img.shape
            self.log(f"成功读取图片: {image_path}")
            self.log(f"图片尺寸: {width} x {height} 像素")

            _, buffer = cv2.imencode('.png', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            if self._base_url == "https://api.mindcraft.com.cn/v1/":
                return img_base64
            else:
                return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            self.log(f"读取图片时出错: {e}")
            return None

    # ==================== JSON 解析 ====================

    def parse_json(self, json_str):
        """解析 AI 输出的 JSON 字符串，处理格式不规范的情况"""
        try:
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]

            json_str = json_str.strip()

            json_pattern = r'\{\s*"(?:[^"\\]|\\.)*"\s*:\s*(?:"(?:[^"\\]|\\.)*"|\d+\.?\d*|true|false|null|\[.*?\]|\{.*?\})\s*(?:,\s*"(?:[^"\\]|\\.)*"\s*:\s*(?:"(?:[^"\\]|\\.)*"|\d+\.?\d*|true|false|null|\[.*?\]|\{.*?\})\s*)*\}'
            json_matches = re.findall(json_pattern, json_str, re.DOTALL)

            if json_matches:
                valid_json = max(json_matches, key=len)
                self.log(f"从AI输出中提取的JSON: {valid_json}")
                return json.loads(valid_json)
            else:
                self.log("正则匹配JSON失败，尝试原始方法")
                first_brace = json_str.find('{')
                last_brace = json_str.rfind('}')

                if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
                    valid_json = json_str[first_brace:last_brace + 1]
                    self.log(f"提取的有效JSON: {valid_json}")
                    return json.loads(valid_json)
                else:
                    return json.loads(json_str)

        except json.JSONDecodeError as e:
            self.log(f"JSON解析错误: {e}")
            try:
                cleaned_str = re.sub(r'^\s*{\s*{\s*', '{', json_str)
                cleaned_str = re.sub(r'\s*}\s*}\s*$', '}', cleaned_str)
                self.log(f"清理后的JSON: {cleaned_str}")
                return json.loads(cleaned_str)
            except json.JSONDecodeError as e2:
                self.log(f"二次解析失败: {e2}")
                return None
        except Exception as e:
            self.log(f"解析过程中发生错误: {e}")
            return None

    # ==================== 鼠标/键盘控制 ====================

    def _get_image_size(self):
        """获取当前截图的宽高"""
        img = cv2.imread(self._input_path)
        if img is not None:
            return img.shape[:2]
        return None, None

    def _notify_coordinate(self, x, y):
        """通知主窗口 AI 输出的坐标"""
        if self._coordinate_callback and 0 <= x <= 100000 and 0 <= y <= 100000:
            try:
                self._coordinate_callback((x, y))
            except Exception as e:
                self.log(f"调用坐标回调函数时出错: {e}")

    def _execute_hotkey(self, type_information):
        """执行快捷键操作"""
        if not type_information:
            self.log("热键操作但未提供快捷键信息")
            return ""

        keys = type_information.split()
        keys = ["win" if key == "meta" else key for key in keys]
        self.log(f"执行热键操作: {'+'.join(keys)}")
        if len(keys) > 0:
            pyautogui.keyDown(keys[0])
            for key in keys[1:]:
                pyautogui.press(key)
            pyautogui.keyUp(keys[0])
        return f"执行热键操作: {'+'.join(keys)}\n"

    def _execute_drag(self, coordinates, scale, img_width, img_height, duration):
        """执行拖拽操作"""
        start_x, start_y = coordinates[0]
        end_x, end_y = coordinates[1]

        start_x, start_y = map_coordinates(start_x, start_y, scale, img_width, img_height)
        self._notify_coordinate(start_x, start_y)

        end_x, end_y = map_coordinates(end_x, end_y, scale, img_width, img_height)
        self._notify_coordinate(end_x, end_y)

        pyautogui.moveTo(start_x, start_y, duration=duration)
        self.log(f"鼠标已移动到拖拽起点: ({start_x}, {start_y})")
        pyautogui.dragTo(end_x, end_y, duration=duration * 10, button='left')
        self.log(f"已完成拖拽操作: ({start_x}, {start_y}) -> ({end_x}, {end_y})")

        return "鼠标已移动到拖拽起点\n已完成拖拽操作\n", [[start_x, start_y], [end_x, end_y]]

    def _execute_single_point_action(self, x, y, action, scale, img_width, img_height, duration):
        """执行单点鼠标操作"""
        x, y = map_coordinates(x, y, scale, img_width, img_height)
        self._notify_coordinate(x, y)

        pyautogui.moveTo(x, y, duration=duration)
        action_str = f"鼠标已移动到坐标\n"

        scroll_range = 10 if self._current_os == "Darwin" else 500

        action_map = {
            "click":         ("已点击",           lambda: pyautogui.click()),
            "double_click":  ("已双击",           lambda: pyautogui.doubleClick()),
            "long_press":    ("已长按",           lambda: pyautogui.mouseDown(button='left')),
            "right_click":   ("已右键点击",       lambda: pyautogui.rightClick()),
            "scroll_up":     (f"已向上滚动 {scroll_range}", lambda: pyautogui.scroll(scroll_range)),
            "scroll_down":   (f"已向下滚动 {scroll_range}", lambda: pyautogui.scroll(-1 * scroll_range)),
        }

        if action in action_map:
            desc, func = action_map[action]
            func()
            self.log(f"{desc} ({x}, {y})")
            action_str += f"{desc}\n"
        else:
            self.log(f"未知操作: {action}，默认执行点击")
            pyautogui.click()
            action_str += "已点击\n"

        return action_str, [x, y]

    def _execute_text_input(self, action, type_information, x, y):
        """处理文本输入"""
        if not type_information or action == "hotkey":
            return ""

        pyperclip.copy(type_information)
        time.sleep(0.1)

        if action == "type_replace":
            pyautogui.click()
            self.log(f"已点击 ({x}, {y})")
            pyautogui.hotkey('ctrl', 'a')

        pyautogui.hotkey('ctrl', 'v')
        self.log(f"已粘贴: {type_information}")
        time.sleep(0.5)
        pyautogui.press('enter')
        time.sleep(0.5)
        self.log("已发送")
        return f"已发送: {type_information}\n"

    def move_mouse_to_coordinates(self, coordinates, solving_problem, action, type_information, scale=1):
        """将鼠标移动到指定坐标并执行相应操作"""

        def _fix_coordinates(coords):
            def _clamp(val):
                return max(-100000, min(100000, val)) if isinstance(val, (int, float)) else val

            if isinstance(coords[0], list):
                if len(coords) == 1:
                    return [_clamp(coords[0][0]), _clamp(coords[0][1])]
                return [
                    [_clamp(coords[0][0]), _clamp(coords[0][1])],
                    [_clamp(coords[1][0]), _clamp(coords[1][1])]
                ]
            return [_clamp(coords[0]), _clamp(coords[1])]

        coordinates = _fix_coordinates(coordinates)
        duration = self._mouse_config["move_duration"]
        img_height, img_width = self._get_image_size()

        mapped_coordinates = None
        action_str = ""

        # 处理页面加载状态
        if action == "page_loading":
            self.log("检测到页面正在加载，暂停0.5秒...")
            time.sleep(0.5)
            self.log("暂停结束，继续操作")
            return "检测到页面正在加载，暂停0.5秒后继续", None

        # 处理热键操作
        if action == "hotkey":
            action_str = self._execute_hotkey(type_information)
            return action_str, None

        # 处理拖拽操作
        if action == "drag" and isinstance(coordinates[0], list):
            action_str, mapped_coordinates = self._execute_drag(
                coordinates, scale, img_width, img_height, duration
            )
        else:
            # 单点操作
            x, y = coordinates
            action_str, mapped_coordinates = self._execute_single_point_action(
                x, y, action, scale, img_width, img_height, duration
            )

        time.sleep(0.2)

        # 处理文本输入
        action_str += self._execute_text_input(action, type_information, coordinates[0], coordinates[1])

        # 如果是做题模式，将鼠标移到左上角避免遮挡
        if solving_problem == "True":
            pyautogui.moveTo(0, 0, duration=duration)

        time.sleep(1.5)
        return action_str, mapped_coordinates

    # ==================== AI 步骤执行 ====================

    def _call_volcano_api(self, user_content, image_data_url):
        """调用火山引擎 API"""
        self.log("小华正在工作中...")
        if self._should_exit:
            return None

        client = self._get_client()
        if client is None:
            return None

        completion = client.beta.chat.completions.parse(
            model=self._model_name,
            messages=[
                {"role": "system", "content": WORKBOT_ROLE},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": user_content}
                ]}
            ],
            response_format=MathResponse,
            extra_body={
                "thinking": {
                    "type": self._thinking_type
                },
            },
        )
        return completion.choices[0].message.content

    def _call_non_volcano_api(self, user_content, image_data_url):
        """调用非火山引擎 API"""
        self.log(f"非火山引擎，模型是{self._model_name}")
        if self._should_exit:
            return None

        client = self._get_client()
        if client is None:
            return None

        completion_raw = client.chat.completions.create(
            model=self._model_name,
            messages=[
                {"role": "system", "content": WORKBOT_ROLE},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": user_content}
                ]}
            ],
        )
        raw_content = completion_raw.choices[0].message.content
        self.log(f"AI 原始返回内容: {raw_content}")
        parsed_json = self.parse_json(raw_content)

        if parsed_json:
            self.log("手动解析成功！")
            return json.dumps(parsed_json, ensure_ascii=False)
        else:
            self.log("手动解析失败，无法处理 AI 返回的内容")
            return None

    def step_work(self, user_content):
        """执行单步 AI 分析工作"""

        if not os.path.exists(self._input_path):
            self.log(f"错误：图片文件不存在 - {os.path.abspath(self._input_path)}")
            return None

        image_data_url = self.read_local_image()
        if not image_data_url:
            self.log("无法继续，图片读取失败")
            return None

        if not self._api_key:
            self.log("\n提示：未设置 API Key，跳过模型分析")
            return None

        self.log("\n小华正在工作...")
        self.fresh_display_info_vl("\n小华正在工作...")

        if self._should_exit:
            self.log("检测到退出标志，跳过API调用")
            return None

        # 确保客户端已创建
        if self._get_client() is None:
            return None

        try:
            if self._is_volcano:
                result = self._call_volcano_api(user_content, image_data_url)
            else:
                result = self._call_non_volcano_api(user_content, image_data_url)
        except Exception as e:
            self.log(f"API调用出错: {e}")
            self.fresh_display_info_vl(f"API调用出错: {e}")
            if self._should_exit:
                self.log("用户主动停止了AI执行")
            else:
                self.log(f"API调用失败: {e}")
                self.fresh_display_info_vl(f"API调用失败: {e}")
            self._global_client = None
            return None

        self.log(result)
        return result

    # ==================== 主控制循环 ====================

    def run(self, user_content, max_iterations=None):
        """
        主控制循环，自动执行 GUI 操作直到任务完成

        :param user_content: 用户任务描述
        :param max_iterations: 最大迭代次数，默认使用配置中的值
        :return: 最终状态描述
        """
        if max_iterations is None:
            max_iterations = self._execution_config["default_max_iterations"]

        self._should_exit = False
        before_output = []
        current_status = "未完成"
        recent_coordinates = []
        same_coordinate_count = 0

        self.log(f"xiaohua start work: {user_content}")

        # 清空 label 目录中的标记图片
        if os.path.exists(self._output_path):
            for filename in os.listdir(self._output_path):
                if filename.startswith("screen_label") and filename.endswith(".png"):
                    file_path = os.path.join(self._output_path, filename)
                    os.remove(file_path)
            self.log(f"已清空label文件夹: {self._output_path}")

        for i in range(max_iterations):
            if self._should_exit:
                self.log("检测到退出标志，停止循环...")
                return "程序已被用户中断"

            self.log(f"\n=================第 {i} 次循环===============")
            start_time = time.time()

            if i == 0:
                before_output = []
                before_content = ""
            else:
                if self._should_exit:
                    self.log("检测到退出标志，停止循环...")
                    return "程序已被用户中断"

                before_output.append(json.dumps(next_element, ensure_ascii=False))
                if len(before_output) > 10:
                    before_output.pop(0)
                before_content = "之前的AI输出操作为: " + "".join(before_output) + "\n之前已完成的操作为:" + action_str

            try:
                if self._should_exit:
                    self.log("检测到退出标志，停止循环...")
                    return "程序已被用户中断"

                success, scale = capture_screen_and_save(
                    save_path=self._input_path,
                    optimize_for_speed=self._screenshot_config["optimize_for_speed"],
                    max_png=self._screenshot_config["max_png"]
                )
                if not success:
                    self.log("屏幕截图保存失败")
                    continue
                self.log(f"屏幕截图已保存为 {os.path.basename(self._input_path)}")

                next_element = self.step_work(before_content + "\n" + user_content)

                if self._should_exit:
                    self.log("检测到退出标志，停止循环...")
                    return "程序已被用户中断"

                if next_element:
                    next_element = self.parse_json(next_element)
                    if not next_element:
                        self.log("错误：无法解析AI响应，跳过本次循环")
                        continue
                    current_status = next_element.get('current_status', '未知状态')
                    solving_problem = next_element.get('solving_problem', 'False')
                    whether_completed = next_element.get('whether_completed', 'difficult')
                    element_info = next_element.get('element_info', '未知元素')
                    coordinates = next_element.get('coordinates', [0, 0])
                    action = next_element.get('action', '未知操作')
                    type_information = next_element.get('type_information', '')

                    self.fresh_display_info_vl(f"当前状态: {current_status}")

                    if whether_completed == "True":
                        self.log(f"小华用时: {time.time() - start_time:.2f}秒")
                        return current_status
                    elif whether_completed == "difficult":
                        self.log(f"小华用时: {time.time() - start_time:.2f}秒")
                        return current_status

                    self.log(f"小华用时: {time.time() - start_time:.2f}秒")
                    self.log(f"下步工作：点击 {element_info}")
                    self.fresh_display_info_vl(f"小华用时: {time.time() - start_time:.2f}秒")
                    self.fresh_display_info_vl(f"下步工作：点击 {element_info}")

                    # 检查坐标是否重复
                    coordinates_match = False
                    for coord in recent_coordinates:
                        if coord[0] == coordinates[0] and coord[1] == coordinates[1]:
                            coordinates_match = True
                            break

                    if not coordinates_match:
                        recent_coordinates.append(coordinates.copy())
                        same_coordinate_count = 1
                        if len(recent_coordinates) > 3:
                            recent_coordinates.pop(0)
                    else:
                        same_coordinate_count += 1

                    if same_coordinate_count >= 3:
                        self.log("检测到连续3次相同坐标，清空记忆")
                        before_output = []
                        same_coordinate_count = 0
                        recent_coordinates = []

                    action_str, mapped_coordinates = self.move_mouse_to_coordinates(
                        coordinates, solving_problem, action, type_information, scale=scale
                    )

                    # 标记坐标点
                    if mapped_coordinates:
                        img = cv2.imread(self._input_path)
                        if img is not None:
                            if isinstance(mapped_coordinates[0], list):
                                image_coordinates = []
                                for coord in mapped_coordinates:
                                    image_coordinates.append([int(coord[0] * scale), int(coord[1] * scale)])
                            else:
                                image_coordinates = [int(mapped_coordinates[0] * scale), int(mapped_coordinates[1] * scale)]

                            output_filename = f"screen_label{i + 1}.png"
                            output_filepath = os.path.join(self._output_path, output_filename)
                            mark_coordinate_on_image(
                                image_coordinates,
                                input_path=self._input_path,
                                output_path=output_filepath
                            )
                else:
                    self.log("错误：未收到模型响应")
            except Exception as e:
                self.log(f"第 {i} 次循环发生错误: {e}")
                raise e

        return current_status