# Process

#准备  
创建input文件夹  
Track1

#训练  
代码中使用了100000行数据进行100轮训练


## Description
- 特征：均为原始特征，不包含多媒体内容特征。使用到的特征字段 ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
       'music_id', 'did',]
- 模型：基于xDeepFM简单修改的多任务模型(没有测过开放性预测的效果，也可能分开做更好)。

## Environment

 python 3.6  
 deepctr==0.3.1 
 tensorflow-gpu(tensorflow)
 pandas
 scikit-learn

### deepctr install Instruction
- CPU版本
  ```bash
  $ pip install deepctr==0.3.1
  ``` 
- GPU版本
  先确保已经在本地安装`tensorflow-gpu`,版本为 **`tensorflow-gpu>=1.4.0,!=1.7.*,!=1.8.*,<=1.12.0`**，然后运行命令
    ```bash
    $ pip install deepctr==0.3.1 --no-deps
    ```


## 运行说明
1. 将track1对应的数据下载并解压至`input`目录内
2. 根据离线测试和线上提交修改`train.py`中的`ONLINE_FLAG`变量，运行`train_ori.py`，`train_dur.py`,`train_pop.py`文件

