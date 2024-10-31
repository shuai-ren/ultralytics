1. 在datasets文件夹下创建新的数据集文件夹，比如sample
2. 将此路径下的所有文件复制到上一步新建的文件夹sample
3. 修改labels.txt，从第三行开始修改标签内容
4. 修改sample.yaml中的path和names
5. 上传zip格式的数据集压缩包到sample文件夹
6. 执行./mkd.sh制作数据集

mkd.sh脚本流程如下
1. 创建temp, images, labels文件夹，其中temp用于解压数据集最终会删除，images路径下生成train文件夹包含所有图像，labels路径下生成train文件夹包含所有标签，如果文件夹存在会清空其中的所有文件
2. 解压zip数据集至当前路径，然后移动所有图像到temp下，最后删除空文件夹
3. labelme格式转为coco格式，生成annotations.json和train文件夹包含所有图像，将train移动到images下
4. coco格式转成yolov8格式
5. 划分出20%验证集
