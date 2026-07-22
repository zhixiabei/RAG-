from __future__ import annotations

import atexit
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
BACKEND_SRC = ROOT / "backend" / "src"
FRONTEND_DIR = ROOT / "frontend"
children: list[subprocess.Popen] = []


def stop_children() -> None:
    if sys.platform == "win32":
        for process in reversed(children):
            if process.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        return
    for process in reversed(children):
        if process.poll() is None:
            process.terminate()
    for process in reversed(children):
        if process.poll() is None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def handle_signal(signum, _frame) -> None:
    stop_children()
    raise SystemExit(0)


def ensure_port_available(port: int, service_name: str) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(0.3)
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"{service_name} 端口 {port} 已被占用，请先停止原来的进程。")


def start_infrastructure() -> None:
    docker = shutil.which("docker")
    if not docker:
        candidates = [
            Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
            Path(r"C:\Program Files\Docker\Docker\resources\bin\com.docker.cli.exe"),
        ]
        docker = next((str(path) for path in candidates if path.exists()), None)
    if not docker:
        raise RuntimeError(
            "找不到 Docker CLI。请安装 Docker Desktop，启动后重启 PyCharm；"
            "也可以在 PyCharm Terminal 执行 docker --version 检查。"
        )
    try:
        subprocess.run([docker, "compose", "up", "-d", "postgres", "qdrant", "minio"], cwd=ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Docker CLI 已找到，但 Docker Desktop 未运行或 Docker 服务不可用。") from exc


def start_backend() -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "rag_app.main:app",
            "--app-dir",
            str(BACKEND_SRC),
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
        ],
        cwd=ROOT,
    )


def start_frontend() -> subprocess.Popen:
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        raise RuntimeError("找不到 npm，请先安装 Node.js。")
    if not (FRONTEND_DIR / "node_modules").exists():
        raise RuntimeError("前端依赖尚未安装，请先在 frontend 目录执行 npm.cmd install。")
    return subprocess.Popen([npm, "run", "dev"], cwd=FRONTEND_DIR)


def main() -> None:
    atexit.register(stop_children)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    ensure_port_available(8080, "后端")
    ensure_port_available(5173, "前端")
    start_infrastructure()
    time.sleep(2)
    backend = start_backend()
    children.append(backend)
    frontend = start_frontend()
    children.append(frontend)

    print("RAG 本机环境已启动")
    print("前端: http://127.0.0.1:5173")
    print("后端: http://127.0.0.1:8080/docs")
    print("按 Ctrl+C 同时停止前后端进程")
    while True:
        if backend.poll() is not None:
            raise RuntimeError(f"后端进程已退出，退出码: {backend.returncode}")
        if frontend.poll() is not None:
            raise RuntimeError(f"前端进程已退出，退出码: {frontend.returncode}")
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"启动失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
