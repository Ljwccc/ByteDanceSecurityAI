# 字节跳动安全AI挑战赛-大佬等等我

## **1. 模型及特征**
- 模型：catboost
- 特征：
    - 用户侧特征：
       - 账户本身的基础特征
       - 账户本身的特征计数统计
       - 粉丝量、关注量、发帖量、被点赞量、最后登陆时间-注册时间 乘除交叉
	   - 从请求数据中提取出来的device_type, app_version, app_channel类别特征，直接作为静态画像使用
	   - 类别特征下的数值统计特征 min/sum/max/std

    - 请求侧特征：
      - 用户请求的时间序列特征, 时间差序列特征 min/sum/max/std
      - w2v特征， 每个用户的请求ip序列建模


## **2. 算法性能**
- 运行环境: window11
- 资源配置：cpu 4800H 16G内存
- 总计耗时：5折约为14分钟，线上分数0.8853