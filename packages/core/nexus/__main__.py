"""允许 `python -m nexus serve ...` 直接启动服务。"""

from nexus.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
