# d011pro_traffic
vue+django 大模型道路交通流量LSTM预测流量|大模型出行建议|图谱可视化系统【加强版】
编号： D011 pro  【2026 版本】
亮点： 大模型出行助手 + 深度学习预测 + 可视化分析

>
完成源码收费，请加QQ 81040295 注明来意
关注B站，有好处！
>

## 视频演示

[演示视频](https://www.bilibili.com/video/BV1txMmzJEtm)

## 技术架构
vue + django +大模型 + mysql + neo4j
大模型：基于大模型结合温度、流量、出行目的、出行方式给出出行建议
预测功能： 基于`keras`的LSTM 道路流量预测算法 
可视化： echarts 、可以根据时间段来进行流量对比、拥堵情况分析等

## 功能清单
1. 智能出行：基于大模型结合天气、气温、出行目的、出行方式、道路流量给出出行建议
2. 全新预测算法：LSTM 道路流量预测算法
3. 数据分析（根据时间段+区域地图来展示交通流量+其他字段分析）
4. 对比分析：可以选择两条路对比预测流量和道路基本信息
5. 知识图谱：交通相关的产业的知识图谱、支持模糊搜索
6. 登录和注册、个人信息修改、密码修改
7. 道路数据的查看、道路数据库支持模糊查询（路名、区域等关键词）
## 技术说明
![在这里插入图片描述](1-技术说明.png)

## 详细介绍
一、智能出行功能
基于大模型结合天气、气温、出行目的、出行方式、道路流量给出出行建议
![在这里插入图片描述](2-出行建议.png)
选择天气、输入气温、选择出行方式、出行目的、选择道路，根据以上的数据结合根据系统的路况情况发送给大模型，大模型会给出出行建议。
![在这里插入图片描述](3-智能出行.png)
![在这里插入图片描述](4-智能出行.png)
![在这里插入图片描述](5-智能出行.png)


二、LSTM 道路流量预测算法
1. **数据驱动**：直接从数据库获取最新路段流量数据
2. **智能切换**：根据数据量自动选择LSTM或线性回归
3. **时间序列处理**：使用滑动窗口(look_back=3)创建训练数据
4. **模型结构**：双层LSTM + Dense输出层
5. **预测输出**：未来3个时段的流量预测值
![在这里插入图片描述](6-交通流量预测亳州.png)
预测加载中的动画：
![在这里插入图片描述](7-流量预测.png)
![在这里插入图片描述](8-流量预测.png)
LSTM 相关的预测代码：
```python
    def create_lstm_model(self, data, look_back=3):
        """创建并训练LSTM模型"""
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))

        # 创建训练数据集
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:(i + look_back), 0])
            y.append(scaled_data[i + look_back, 0])

        X = np.array(X)
        y = np.array(y)

        # 重塑数据为LSTM输入格式 [样本数, 时间步长, 特征数]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # 创建LSTM模型
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # 训练模型
        model.fit(X, y, epochs=100, batch_size=1, verbose=0)

        return model, scaler

    def lstm_predict(self, model, scaler, data, look_back=3, steps=3):
        """使用LSTM模型进行预测"""
        # 准备输入数据
        scaled_data = scaler.transform(data.reshape(-1, 1))
        x_input = scaled_data[-look_back:].reshape(1, look_back, 1)

        # 进行多步预测
        predictions = []
        for _ in range(steps):
            pred = model.predict(x_input, verbose=0)
            predictions.append(pred[0, 0])

            # 更新输入数据
            x_input = np.append(x_input[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        # 反归一化预测结果
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten().astype(int)
```

三、数据分析（根据时间段+区域地图来展示交通流量+其他字段分析）
先看一个整体效果
![在这里插入图片描述](9-数据分析.png)
![在这里插入图片描述](10-数据分析.png)

这边做了多个图形，并且可以选择时间段（右上角）来进行分析
![在这里插入图片描述](11-地图分析.png)
![在这里插入图片描述](12-地图分析.png)
![在这里插入图片描述](13-选择时间段.png)

![在这里插入图片描述](14-拥堵排名.png)
滑块可以拖动：
![在这里插入图片描述](15-拥堵排名.png)

四、对比分析：可以选择两条路对比预测流量和道路基本信息
![在这里插入图片描述](16-对比分析.png)
![在这里插入图片描述](17-对比分析.png)
![在这里插入图片描述](18-LSTM预测.png)

五、知识图谱：交通相关的产业的知识图谱、支持模糊搜索
![在这里插入图片描述](19-图谱预测.png)

六、登录和注册、个人信息修改、密码修改
登录页面
![在这里插入图片描述](20-登录.png)
修改头像和个人信息
![在这里插入图片描述](21-修改头像.png)
修改密码
![在这里插入图片描述](22-修改密码.png)

七、道路数据的查看、道路数据库支持模糊查询（路名、区域等关键词）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e966c4d21cd843b2b37e9caa3986bd18.png)

