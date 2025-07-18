#!/bin/bash
source ~/.bashrc
# 指定要遍历的目录
INPUT_DIR="/home/yssun/pytorch-fm/torchfm/model/test_fx/exp/model_repo/onn_mlp"
# 指定要执行的命令的路径（假设命令可以接受文件作为参数）
COMMAND="trtexec"
# 指定输出日志的目录
LOG_DIR="/home/yssun/pytorch-fm/torchfm/model/test_fx/exp/log_repo/onn_mlp_gpu"
# 检查日志目录是否存在，不存在则创建
mkdir -p "$LOG_DIR"
export PATH=$PATH:/usr/local/cuda/bin:/home/yssun/TensorRT-10.8.0.43/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/yssun/TensorRT-10.8.0.43/lib
# 遍历指定目录下的所有.onnx文件
for FILE in "$INPUT_DIR"/*.onnx; do
    # 检查是否为文件
    if [ -f "$FILE" ]; then
        # 获取文件名，不含路径，同时移除.onnx扩展名
        BASENAME=$(basename "$FILE" .onnx)
        # 构建日志文件路径
        LOG_FILE="$LOG_DIR/$BASENAME.log"
        # 执行命令并将输出重定向到日志文件
        "$COMMAND"   --onnx="$FILE" --iterations=100 --dumpProfile --dumpLayerInfo --exportProfile=profile.json --exportLayerInfo=layer.json --verbose > "$LOG_FILE" 2>&1
    fi
done

echo "所有.onnx文件处理完成。"
