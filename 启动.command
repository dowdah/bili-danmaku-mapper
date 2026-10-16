#!/bin/zsh
set -eu
cd -- "${0:A:h}"
export PATH="$PWD/tools/ffmpeg/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import sys;sys.exit(sys.version_info < (3,11))'; then
  .venv/bin/python launcher.py
elif command -v python3 >/dev/null && python3 -c 'import sys;sys.exit(sys.version_info < (3,11))'; then
  python3 launcher.py
else
  /usr/bin/osascript -e 'display dialog "需要 Python 3.11 或更新版本。请先从 Python 官网安装，再双击启动。" buttons {"打开安装页面"} default button 1'
  /usr/bin/open 'https://www.python.org/downloads/macos/'
fi
