# -*- coding: utf-8 -*-
"""CatPaw Process Monitor — AMINQT A-Share Quant Platform.

Monitors FastAPI (uvicorn) and Streamlit processes, auto-restarts on failure.
Adapted from EWM Robot project monitor_catpaw.py.

Usage:
    python monitor_catpaw.py              # continuous monitoring
    python monitor_catpaw.py --once        # one-time health check
    python monitor_catpaw.py --status      # status report
    python monitor_catpaw.py --create-task # create Windows Task Scheduler XML
"""
import json, logging, os, subprocess, sys, threading, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    print("psutil not installed. Run: pip install psutil requests")
    sys.exit(1)


class CatPawMonitor:
    """Monitor and auto-restart AMINQT processes."""

    def __init__(self, config_file: str = "catpaw_monitor_config.json"):
        self.config = self._load_config(config_file)
        self._setup_logging()
        self.processes: dict[str, dict[str, Any]] = {}
        self.stop_event = threading.Event()

    def _load_config(self, config_file: str) -> dict[str, Any]:
        default = {
            "processes": [
                {"name": "aminqt_api", "command": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
                 "working_dir": r"D:\AMINQT\AMINQT CODES", "health_check_url": "http://localhost:8000/docs",
                 "health_check_timeout": 5, "restart_delay": 10, "max_restarts_per_hour": 5},
                {"name": "aminqt_streamlit", "command": "streamlit run app/streamlit_app.py --server.port 8501",
                 "working_dir": r"D:\AMINQT\AMINQT CODES", "health_check_url": "http://localhost:8501",
                 "health_check_timeout": 5, "restart_delay": 15, "max_restarts_per_hour": 3},
            ],
            "monitoring": {"check_interval": 30, "log_level": "INFO", "log_file": "catpaw_monitor.log",
                           "max_log_size_mb": 10, "backup_count": 3, "enable_http_check": True},
        }
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    return {**default, **json.load(f)}
            except Exception as e:
                print(f"Config load error: {e}, using defaults")
                return default
        else:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=2, ensure_ascii=False)
            print(f"Created default config at {config_path}")
            return default

    def _setup_logging(self):
        log_config = self.config["monitoring"]
        logging.basicConfig(
            level=getattr(logging, log_config.get("log_level", "INFO")),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_config.get("log_file", "catpaw_monitor.log"), encoding="utf-8"),
                      logging.StreamHandler(sys.stdout)])
        self.logger = logging.getLogger("CatPawMonitor")

    def _is_process_running(self, name: str, pid: int | None = None) -> bool:
        try:
            if pid: psutil.Process(pid); return True
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if name in " ".join(proc.info["cmdline"] or []): return True
                except (psutil.NoSuchProcess, psutil.AccessDenied): continue
            return False
        except psutil.NoSuchProcess: return False
        except Exception as e:
            self.logger.error("Error checking %s: %s", name, e); return False

    def _check_http_health(self, url: str, timeout: int) -> bool:
        try:
            import requests
            return requests.get(url, timeout=timeout).status_code == 200
        except Exception as e:
            self.logger.debug("HTTP health failed %s: %s", url, e); return False

    def _start_process(self, cfg: dict[str, Any]) -> subprocess.Popen | None:
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(r"D:\AMINQT\AMINQT CODES"))
            self.logger.info("Starting %s: %s", cfg["name"], cfg["command"])
            proc = subprocess.Popen(cfg["command"], shell=True, cwd=cfg.get("working_dir"),
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            threading.Thread(target=self._read_output, args=(cfg["name"], proc.stdout, "STDOUT"), daemon=True).start()
            threading.Thread(target=self._read_output, args=(cfg["name"], proc.stderr, "STDERR"), daemon=True).start()
            return proc
        except Exception as e:
            self.logger.error("Failed to start %s: %s", cfg["name"], e); return None

    def _read_output(self, name: str, pipe, stream: str):
        try:
            for line in iter(pipe.readline, ""):
                if line.strip(): self.logger.info("[%s %s] %s", name, stream, line.rstrip())
        except Exception as e: self.logger.debug("Read error %s: %s", stream, e)

    def _monitor_process(self, cfg: dict[str, Any]):
        name = cfg["name"]
        restart_history: list[datetime] = []
        while not self.stop_event.is_set():
            try:
                if name not in self.processes:
                    proc = self._start_process(cfg)
                    if proc:
                        self.processes[name] = {"process": proc, "pid": proc.pid,
                            "start_time": datetime.now(), "restart_count": 0}
                    else: time.sleep(cfg["restart_delay"]); continue
                info = self.processes[name]
                rc = info["process"].poll()
                if rc is not None:
                    self.logger.warning("%s (PID %s) exited code %s", name, info["pid"], rc)
                    now = datetime.now()
                    recent = [t for t in restart_history if t > now - timedelta(hours=1)]
                    if len(recent) >= cfg["max_restarts_per_hour"]:
                        self.logger.error("%s exceeded restart limit. Not restarting.", name)
                        time.sleep(60); continue
                    time.sleep(cfg["restart_delay"])
                    new_proc = self._start_process(cfg)
                    if new_proc:
                        self.processes[name] = {"process": new_proc, "pid": new_proc.pid,
                            "start_time": now, "restart_count": info["restart_count"] + 1}
                        restart_history.append(now)
                        restart_history = [t for t in restart_history if t > now - timedelta(hours=24)]
                    else: self.logger.error("Failed to restart %s", name)
                if self.config["monitoring"].get("enable_http_check"):
                    url = cfg.get("health_check_url")
                    if url and not self._check_http_health(url, cfg["health_check_timeout"]):
                        self.logger.warning("Health check failed for %s at %s", name, url)
            except Exception as e: self.logger.error("Monitor error %s: %s", name, e)
            time.sleep(self.config["monitoring"]["check_interval"])

    def stop_all(self):
        self.logger.info("Stopping all processes...")
        self.stop_event.set()
        for name, info in self.processes.items():
            try:
                proc = info["process"]
                if proc.poll() is None:
                    self.logger.info("Stopping %s (PID %s)", name, info["pid"])
                    proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill()
            except Exception as e: self.logger.error("Stop error %s: %s", name, e)

    def _log_status(self):
        lines = ["=== CatPaw Monitor Status (AMINQT) ==="]
        for name, info in self.processes.items():
            proc = info["process"]
            uptime = str(datetime.now() - info["start_time"]).split(".")[0]
            status = "RUNNING" if proc.poll() is None else f"STOPPED (code: {proc.poll()})"
            lines.append(f"{name}: {status} | PID: {info['pid']} | Uptime: {uptime} | Restarts: {info['restart_count']}")
        self.logger.info("\n".join(lines))

    def run(self):
        self.logger.info("Starting CatPaw Monitor (AMINQT)...")
        for cfg in self.config["processes"]:
            threading.Thread(target=self._monitor_process, args=(cfg,), daemon=True).start()
            self.logger.info("Started monitor for %s", cfg["name"])
        try:
            while not self.stop_event.is_set(): time.sleep(300); self._log_status()
        except KeyboardInterrupt: self.logger.info("Keyboard interrupt")
        finally: self.stop_all(); self.logger.info("Monitor stopped")


def _create_windows_task_xml():
    xml = '''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Auto-start CatPaw Monitor (AMINQT)</Description></RegistrationInfo>
  <Triggers><BootTrigger><Enabled>true</Enabled></BootTrigger><LogonTrigger><Enabled>true</Enabled></LogonTrigger></Triggers>
  <Principals><Principal id="Author"><UserId>S-1-5-18</UserId><RunLevel>HighestAvailable</RunLevel></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy><Enabled>true</Enabled><Priority>7</Priority></Settings>
  <Actions Context="Author"><Exec><Command>python</Command><Arguments>"D:\\AMINQT\\AMINQT CODES\\monitor_catpaw.py"</Arguments><WorkingDirectory>D:\\AMINQT\\AMINQT CODES</WorkingDirectory></Exec></Actions>
</Task>'''
    xml_path = Path(r"D:\AMINQT\AMINQT CODES") / "catpaw_monitor_task.xml"
    with open(xml_path, "w", encoding="utf-8") as f: f.write(xml)
    print(f"Created: {xml_path}")
    print(f'Register (Admin PowerShell): schtasks /create /xml "{xml_path}" /tn "AMINQTMonitor"')


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="AMINQT CatPaw Process Monitor")
    p.add_argument("--create-task", action="store_true")
    p.add_argument("--config", default="catpaw_monitor_config.json")
    p.add_argument("--once", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.create_task: _create_windows_task_xml()
    elif args.status: CatPawMonitor(args.config)._log_status()
    elif args.check:
        m = CatPawMonitor(args.config)
        for c in m.config["processes"]:
            print(f"\n=== {c['name']} ===")
            print(f"  Process: {m._is_process_running(c['name'])}")
            if c.get("health_check_url"): print(f"  Health: {m._check_http_health(c['health_check_url'], c['health_check_timeout'])}")
    elif args.once:
        m = CatPawMonitor(args.config)
        for c in m.config["processes"]:
            print(f"\n{c['name']}: Process={m._is_process_running(c['name'])}", end="")
            if c.get("health_check_url"): print(f", Health={m._check_http_health(c['health_check_url'], c['health_check_timeout'])}")
    else: CatPawMonitor(args.config).run()
