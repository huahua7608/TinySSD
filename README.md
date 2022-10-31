# TinySSD
Homework of AI Experiment

## 环境配置

- CUDA == 4.5.11([Anaconda](https://www.anaconda.com/)或[Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links))
- Python == 3.8 
- [PyTorch](https://pytorch.org/) == 1.11.0

#### 环境配置步骤

1. 下载代码和数据集文件

    ```bash
    
     ├─TinySSD-main
      │  
      ├─detection
      │          
      ├─results
      │    
      ├─utils
      │  
      ├─weights
      │          
      ├─model.py
      │          
      ├─train.py
      │      
      └─test.py
    ```

2. 安装环境([Anaconda](https://www.anaconda.com/)或[Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links))
    ```bash
    点击跳转网页直接下载安装即可
    ```


3.  运行create_train.py生成训练集
    运行完成后detection文件夹目录下多了一个sysu_train的子文件夹：
    ```     
      ├─sysu_train
         ├─label.csv
         │  
         └─images
    ```
## 训练流程

#### 训练
   - 运行train.py即可训练，可在代码中直接修改batch_size和epoch
    `
    python train.py
    `              
#### 测试
   - 运行test.py即可测试，注意test数据集路径和weight文件的路径必须与代码保持一致，测试结果将保存在results文件夹下
    `
    python test.py
    `
    

