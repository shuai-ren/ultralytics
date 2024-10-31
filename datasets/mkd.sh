#!/bin/bash

# 目录名称
dirs=("temp" "images" "labels")

# 遍历目录名称数组
for dir in "${dirs[@]}"; do
  # 检查目录是否存在
  if [ -d "$dir" ]; then
    echo "路径 $dir 存在. 已删除..."
    rm -rf "$dir"  # 删除目录及其内容
  fi

  # 创建新目录
  echo "创建 $dir..."
  mkdir "$dir"
done

if [ -d "train" ]; then
  echo "路径 train 存在. 已删除..."
  rm -rf "train"  # 删除目录及其内容
fi

# 目标文件夹路径
destination="temp"

# 遍历当前目录中的所有zip文件
for file in *.zip; do
    # 创建与zip文件同名的目录
    unzip -q "$file"
    
    # 移动所有文件到目标文件夹
    mv "${file%.zip}"/* "$destination"
    
    # 删除空文件夹
    rmdir "${file%.zip}"
done

echo "解压缩、移动和删除空文件夹完成！"

python3 labelme2coco.py $destination ./ --labels=labels.txt

rm -r $destination
mv train images

python3 general_json2yolo.py

#rm annotations.json
python3 create_val_set.py

echo "数据集制作完成."
