# 环境配置
安装库geohash2,pytorch,sklearn,pandas,numpy

# 运行方式
在config.py文件中配置训练文件和测试文件路径，然后执行python GRU_poi.py即可

# 数据集
目前支持

foursquare_nyc/specific_train.csv, foursquare_nyc/specific_test.csv

foursquare_global/specific_train.csv, foursquare_global/specific_test.csv

brightkite/train.csv, brightkite/test.csv

gowalla/train.csv, gowalla/test.csv

这四个数据集

# 模型

采用GRU对poi序列进行建模，轨迹长度对齐方法采用最大长度对齐（padding补0时，从起始位置开始补零）