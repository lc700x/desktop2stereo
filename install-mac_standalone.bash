 #!/bin/bash
cd "$(dirname "$0")"
 ./Python310/bin/pip3 install -r requirements-mps.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/
 ./Python310/bin/pip3 install -r requirements.txt --no-cache-dir --trusted-host http://mirrors.aliyun.com/pypi/simple/