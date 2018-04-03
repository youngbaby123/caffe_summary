作者：杨小杭
日期：2018年4月3日

## Caffe 图像检测 Summary 代码

---
#### 文件1：Get_img_list.py
    
    图片预处理，用来生成label.txt, train.txt, val.txt, test.txt

#### 文件2：get_label_dir.py

    按照指定几级子目录进行文件转存

#### 文件3：create_lmdb.sh
    
    lmdb文件生成
    
#### 文件4：test_.py
    单张图片测试类
    
#### 文件5：test_demo.py

    test_.py的调用demo

#### 文件6：summary_.py

    对检测结果进行批量统计

#### 文件7：summary_demo.py

    summary_.py的调用demo
    
#### 文件8：summary_test_all.py

    对label转存后的检测结果进行批量统计
    
#### 文件9：r_label_to_label.py
    
    检测label与实际label不一致时, 进行label转存
    
#### 文件10：classifi_more_data.py

    直接对图片进行检测, 不存在test.txt, 并且会根据检测结果进行转存
