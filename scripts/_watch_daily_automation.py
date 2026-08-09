"""监控 daily_automation 日志: 步骤开始/完成 + OOM/失败信号 (GBK 解码)."""
import time

PATH = r"logs/daily_automation_20260809.log"
KEYS = (
    "[start]", "[done]", "[stale]", "MemoryError", "Unable to allocate",
    "allocate", "Traceback", "ERROR", "Error", "FAIL", "失败", "完成",
    "成功", "switched", "current_meta", "OOS", "publish", "交付",
)


def main() -> None:
    last = 0
    while True:
        try:
            with open(PATH, "rb") as fp:
                fp.seek(last)
                chunk = fp.read()
        except FileNotFoundError:
            chunk = b""
        if chunk:
            last += len(chunk)
            text = chunk.decode("gbk", errors="replace")
            for line in text.splitlines():
                low = line.lower()
                if any(k in low for k in KEYS):
                    print(line.strip(), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
