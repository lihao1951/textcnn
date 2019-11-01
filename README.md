# textcnn
textcnn-flask-api

classification.py：训练textcnn模型

inference.py：flask-api文件

textcnn.ini：uwsgi配置文件

yq_test.txt  yq_train.txt  yq_valid.txt：从网络搜索得到的数据，分类不够精准，测试用
 
rcnn.py：测试rcnn模型文件 经训练后发现和textcnn精度相差不大
 
utils.py：文本处理文件

config：停用词、自定义词、词典

templates：含有测试页面

经测试数据：textcnn的分类准确度稳定在0.87左右
