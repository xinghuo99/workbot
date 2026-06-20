"""workbot 主入口"""

import time
from workbot import WorkBot


def main():
    bot = WorkBot()
    bot.log("=== 本地图片分析工具 ===")
    bot.log("按 Ctrl+C 可以随时退出程序")

    user_content = input("请输入您的需求：")
    time.sleep(3)
    time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    user_content = f"当前时间为:{time_str}\n用户任务为:{user_content}"
    bot.log("正在处理...")
    bot.log(user_content)

    current_time = time.time()
    bot.run(user_content, max_iterations=bot._execution_config["max_visual_model_iterations"])
    bot.log(f"处理时间: {time.time() - current_time} 秒")

    if bot._should_exit:
        bot.log("程序已成功退出")


if __name__ == "__main__":
    main()