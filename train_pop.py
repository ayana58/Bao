import pandas as pd
from deepctr import SingleFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from model_new import xDeepFM_MTL
from auc_util import auroc
from tensorflow.keras.callbacks import Callback, EarlyStopping
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from geneticalgorithm import geneticalgorithm as ga
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random


ONLINE_FLAG = False
loss_weights = [0.7, 0.3, ]  # [0.7,0.3]任务权重可以调下试试
VALIDATION_FRAC = 0.2  # 用做线下验证数据比例

VERSION=4

# 定义模糊隶属函数和模糊系统
# 定义模糊隶属函数和模糊系统
def triangular_membership(x, a, b, c):
    """
    三角形隶属函数，用于模糊化输入值。
    :param x: 输入值
    :param a: 三角形左端点
    :param b: 三角形顶点（最大值为1）
    :param c: 三角形右端点
    :return: 隶属度（介于0到1之间）
    """
    if x <= a or x >= c:
        return 0.0  # 如果x不在a到c之间，隶属度为0
    elif a < x < b:
        return (x - a) / (b - a)  # 在a到b之间，隶属度线性增加
    elif b < x < c:
        return (c - x) / (c - b)  # 在b到c之间，隶属度线性减少
    else:
        return 1.0  # 在x等于b时，隶属度为1

def interest_membership(like_count, total_count):
    """
    计算兴趣隶属度，基于用户点赞数与观看数的比率。
    :param like_count: 用户对某个视频的点赞次数
    :param total_count: 用户观看视频的总次数
    :return: 兴趣隶属度（介于0到1之间）
    """
    if total_count == 0:
        return 0.0  # 如果没有观看过，则兴趣为0
    return like_count / total_count  # 否则，兴趣隶属度为点赞次数除以观看次数

def calculate_loss(recommendation_system, data):
    """
    计算在当前模糊系统配置下的损失（误差）。
    :param recommendation_system: 当前配置的模糊推理系统
    :param data: 包含真实标签的数据
    :return: 计算得到的损失值
    """
    total_loss = 0
    n_samples = len(data)
    
    for index, row in data.iterrows():
        # 为模糊系统输入值
        recommendation_system.input['watch_time'] = row['watch_time_ratio']
        recommendation_system.input['interest_level'] = row['interest_membership']
        
        # 计算模糊系统的输出
        recommendation_system.compute()
        predicted_value = recommendation_system.output['recommendation']
        
        # 计算与真实值之间的误差 (例如均方误差)
        actual_value = row['finish'] 
        total_loss += (predicted_value - actual_value) ** 2
    
    # 返回平均损失
    return total_loss / n_samples


# 利用遗传算法优化隶属函数参数
def optimize_membership_function():
    def objective_function(params):
        a, b, c = params
        watch_time['medium'] = fuzz.trimf(watch_time.universe, [a, b, c])
        loss = calculate_loss(recommendation_system, data)
        return loss

    varbound = np.array([[0, 0.5], [0.5, 0.8], [0.8, 1]])
    algorithm_param = {'max_num_iteration': 100,
                       'population_size': 50,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.3,
                       'crossover_type':'uniform',
                       'max_iteration_without_improv': None}

    model = ga(function=objective_function, dimension=3, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run()
    print('22222222222222222222222222222222222222222222222222222222222')
    best_params = model.output_dict['variable']
    print(f"Optimized membership function parameters: a={best_params[0]}, b={best_params[1]}, c={best_params[2]}")
    return best_params



#添加city的隶属度函数研究
def calculate_city_popularity(city_counts, total_count):
    """
    根据城市用户数量计算城市的流行度隶属度
    :param city_counts: 城市的用户数量
    :param total_count: 总用户数量
    :return: 流行度隶属度值（介于0到1之间）
    """
    return city_counts / total_count

# 定义user_city和item_city的隶属度函数
def add_city_fuzzy_features(data):
    """
    为数据集添加城市流行度模糊特征，包括 user_city 和 item_city 的隶属度。
    """
    # 将 -1 的 city 值处理为 NaN
    data['user_city'].replace(-1, np.nan, inplace=True)
    data['item_city'].replace(-1, np.nan, inplace=True)
    
    # 计算每个城市的用户数和视频发布数
    user_city_counts = data['user_city'].value_counts()
    item_city_counts = data['item_city'].value_counts()
    
    total_users = len(data['uid'].unique())
    total_items = len(data['item_id'].unique())
    
    # 处理 NaN 值，使用默认流行度值0.5
    data['user_city_popularity'] = data['user_city'].apply(
        lambda x: calculate_city_popularity(user_city_counts[x], total_users) if pd.notna(x) else 0.5)
    data['item_city_popularity'] = data['item_city'].apply(
        lambda x: calculate_city_popularity(item_city_counts[x], total_items) if pd.notna(x) else 0.5)
    
    return data

# 使用 skfuzzy库 定义模糊变量和规则
watch_time = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'watch_time')
interest_level = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'interest_level')
recommendation = ctrl.Consequent(np.arange(0, 101, 1), 'recommendation')

# 定义 user_city_popularity 和 item_city_popularity 的模糊变量及隶属度函数
user_city_popularity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'user_city_popularity')
item_city_popularity = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'item_city_popularity')

user_city_popularity['low'] = fuzz.trimf(user_city_popularity.universe, [0, 0, 0.3])
user_city_popularity['medium'] = fuzz.trimf(user_city_popularity.universe, [0.3, 0.5, 0.8])
user_city_popularity['high'] = fuzz.trimf(user_city_popularity.universe, [0.8, 1, 1])

item_city_popularity['low'] = fuzz.trimf(item_city_popularity.universe, [0, 0, 0.3])
item_city_popularity['medium'] = fuzz.trimf(item_city_popularity.universe, [0.3, 0.5, 0.8])
item_city_popularity['high'] = fuzz.trimf(item_city_popularity.universe, [0.8, 1, 1])

# 定义watch_time的隶属度函数
watch_time['low'] = fuzz.trimf(watch_time.universe, [0, 0, 0.3])
watch_time['medium'] = fuzz.trimf(watch_time.universe, [0.3, 0.5, 0.8])
watch_time['high'] = fuzz.trimf(watch_time.universe, [0.8, 1, 1])

# 定义interest_level的隶属度函数
interest_level['low'] = fuzz.trimf(interest_level.universe, [0, 0, 0.3])
interest_level['medium'] = fuzz.trimf(interest_level.universe, [0.3, 0.5, 0.8])
interest_level['high'] = fuzz.trimf(interest_level.universe, [0.8, 1, 1])

# 定义recommendation的隶属度函数
recommendation['low'] = fuzz.trimf(recommendation.universe, [0, 0, 50])
recommendation['medium'] = fuzz.trimf(recommendation.universe, [30, 50, 70])
recommendation['high'] = fuzz.trimf(recommendation.universe, [50, 100, 100])

# 定义模糊规则
rule1 = ctrl.Rule(watch_time['high'] & interest_level['high'], recommendation['high'])
rule2 = ctrl.Rule(watch_time['medium'] & interest_level['medium'], recommendation['medium'])
rule3 = ctrl.Rule(watch_time['low'] | interest_level['low'], recommendation['low'])
# 添加新的规则，将城市的流行度考虑进去
rule4 = ctrl.Rule(watch_time['high'] & interest_level['high'] & user_city_popularity['high'] & item_city_popularity['high'], recommendation['high'])
rule5 = ctrl.Rule(watch_time['medium'] & interest_level['medium'] & user_city_popularity['medium'] & item_city_popularity['medium'], recommendation['medium'])
rule6 = ctrl.Rule(watch_time['low'] | interest_level['low'] | user_city_popularity['low'] | item_city_popularity['low'], recommendation['low'])

rule_default = ctrl.Rule(watch_time['medium'] & interest_level['medium'], recommendation['medium'])

# 创建模糊控制系统
recommendation_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule_default])
recommendation_system = ctrl.ControlSystemSimulation(recommendation_ctrl)

# 将模糊特征集成到训练数据中
def add_fuzzy_features(data):
    """
    将模糊特征集成到训练数据中，生成推荐值。
    """
    #n_samples = len(data)
    #watch_time_ratio = np.random.rand(n_samples)  # 随机生成观看时长比例
    #data['watch_time_ratio'] = watch_time_ratio  # 估算观看时长比例

    # 获取数据集中 video_duration 的最大值
    max_video_duration = data['video_duration'].max()

    # 使用最大值对每个视频时长进行归一化
    data['video_duration_normalized'] = data['video_duration'] / max_video_duration
    data['video_duration_normalized'].fillna(0, inplace=True)

    data['watch_membership'] = data['video_duration_normalized'].apply(lambda x: triangular_membership(x, 0.3, 1.0, 0.8))

    # 计算用户兴趣隶属度
    user_like_counts = data.groupby('uid')['like'].sum()
    user_total_counts = data.groupby('uid')['finish'].count()
    data['interest_membership'] = data['uid'].apply(lambda uid: interest_membership(user_like_counts[uid], user_total_counts[uid]))

    # 添加城市流行度模糊特征
    data = add_city_fuzzy_features(data)

    # 调用模糊推理系统，生成推荐值
    recommendations = []
    for index, row in data.iterrows():
        recommendation_system.input['user_city_popularity'] = row['user_city_popularity']
        recommendation_system.input['item_city_popularity'] = row['item_city_popularity']
        recommendation_system.input['watch_time'] = row['watch_membership']
        recommendation_system.input['interest_level'] = row['interest_membership']

        try:
            recommendation_system.compute()
            recommendations.append(recommendation_system.output['recommendation'])
        except ValueError as e:
            print(f"Error processing row {index}: {e}")
            recommendations.append(np.nan)  # 在遇到错误时返回 NaN 或其他默认值
    
    data['fuzzy_recommendation'] = recommendations  # 将生成的推荐值添加到数据中
    
    return data

if __name__ == "__main__":
    # 读取训练数据集的前1000000行数据，每次读取1000行，以块的形式逐步处理
    data = pd.read_csv('./input/final_track2_train.txt', sep='\t', names=[
        'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'], iterator=True,nrows=1000)
    
    take = []
    loop = True
    while loop:
        try:
            chunk = data.get_chunk(1000)  # 每次读取1000行数据
            chunk = add_fuzzy_features(chunk)  # 对每块数据添加模糊特征
            take.append(chunk)  # 将处理后的块添加到列表中
        except StopIteration:
            loop = False  # 读取完所有数据后停止循环
            print('Stop iteration')
    # 将所有块组合成一个完整的数据集
    data = pd.concat(take, ignore_index=True)
    print(data.shape[0])
    print("Finish labels after loading data:", data['finish'].unique())
    if ONLINE_FLAG:
        # 如果是线上模式，读取测试数据，并将其附加到训练数据集后
        test_data = pd.read_csv('./input/final_track2_test_no_anwser.txt', sep='\t', names=[
                                'uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id', 'did', 'creat_time', 'video_duration'])
        train_size = data.shape[0]  # 记录训练数据集的大小
        data = data.append(test_data)  # 合并训练数据和测试数据
    else:
        # 如果是离线模式，按照验证集比例划分训练集
        train_size = int(data.shape[0]*(1-VALIDATION_FRAC))

    #print('11111111111111111111111111111111111111111111 1')
    # 定义稀疏特征和稠密特征的列表
    sparse_features = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel',
                       'music_id', 'did', ]
    dense_features = ['video_duration']
    dense_features = ['video_duration', 'watch_membership', 'interest_membership', 'user_city_popularity', 'item_city_popularity', 'fuzzy_recommendation']

    data[sparse_features] = data[sparse_features].fillna('-1', )# 对稀疏特征进行缺失值填充，使用 '-1' 填充
    data[dense_features] = data[dense_features].fillna(0,)# 对稠密特征进行缺失值填充，使用 '0' 填充

    target = ['finish', 'like']  #目标标签，分别表示用户是否完成观看和是否点赞。
    print("Finish labels after loading test data:", data['finish'].unique())
    #print('11111111111111111111111111111111111111111111 1')
    # 对稀疏特征进行标签编码
    for feat in sparse_features:
        data[feat] = data[feat].astype(str)
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    # 对稠密特征进行归一化处理
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    # 创建稀疏和稠密特征列表，用于模型输入
    sparse_feature_list = [SingleFeat(feat, data[feat].nunique())
                           for feat in sparse_features]
    dense_feature_list = [SingleFeat(feat, 0)
                          for feat in dense_features]
    # 将数据集分为训练集和测试集
    train = data.iloc[:train_size]
    test = data.iloc[train_size:]
    #print('11111111111111111111111111111111111111111111 1')
    # 准备模型输入数据
    train_model_input = [train[feat.name].values for feat in sparse_feature_list] + \
        [train[feat.name].values for feat in dense_feature_list]
    test_model_input = [test[feat.name].values for feat in sparse_feature_list] + \
        [test[feat.name].values for feat in dense_feature_list]
    # 提取目标标签
    train_labels = [train[target[0]].values, train[target[1]].values]
    test_labels = [test[target[0]].values, test[target[1]].values]
    #print(f"train_model_input contains {len(train_model_input)} arrays.")
    #for i, array in enumerate(train_model_input):
        #print(f"Array {i} shape: {array.shape}")
    # 设置模型参数
    embedding_size=8
    hidden_size=(512, 512)
    if VERSION==3:
        embedding_size=1
    if VERSION==4:
        embedding_size=1
        hidden_size=(64,64)

    # 构建并编译模型
    model = xDeepFM_MTL({"sparse": sparse_feature_list,
                         "dense": dense_feature_list}, embedding_size=embedding_size, hidden_size=hidden_size)
    model.compile("adagrad", loss='binary_crossentropy', loss_weights=loss_weights, metrics=[auroc])
    # 定义早停回调函数
    my_callbacks = [EarlyStopping(monitor='loss', min_delta=1e-2, patience=10, verbose=1, mode='min')]
    print('11111111111111111111111111111111111111111111 1')
    print("Finish labels after loading data:", np.unique(test_labels[0]))

    # 开始训练模型
    if ONLINE_FLAG:
        # 线上模式下，不进行验证，直接训练模型
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=10, verbose=1,
                            callbacks=my_callbacks)
        pred_ans = model.predict(test_model_input, batch_size=256)

    else:
        # 离线模式下，使用验证集进行训练
        history = model.fit(train_model_input, train_labels,
                            batch_size=4096, epochs=10, verbose=1, validation_data=(test_model_input, test_labels),
                            callbacks=my_callbacks)

    if ONLINE_FLAG:
        # 如果是线上模式，生成并保存预测结果
        result = test_data[['uid', 'item_id', 'finish', 'like']].copy()
        result.rename(columns={'finish': 'finish_probability',
                               'like': 'like_probability'}, inplace=True)
        result['finish_probability'] = pred_ans[0]
        result['like_probability'] = pred_ans[1]
        result[['uid', 'item_id', 'finish_probability', 'like_probability']].to_csv('result.csv', index=None, float_format='%.6f')

    # Evaluate the model
    pred_ans = model.predict(test_model_input, batch_size=2**14)
    roc_auc_fuzzy = roc_auc_score(test_labels[0], test['fuzzy_recommendation'])
    print("Test labels (finish):", test_labels[0])
    print("Minimum value in test_labels[0]:", np.min(test_labels[0]))
    print("Minimum value in test_labels[0]:", np.max(test_labels[0]))
    print("Finish label distribution:", np.bincount(test_labels[0]))
    print("Like label distribution:", np.bincount(test_labels[1]))
    roc_auc_finish = roc_auc_score(test_labels[0], pred_ans[0])
    roc_auc_like = roc_auc_score(test_labels[1], pred_ans[1])

    accuracy_finish = accuracy_score(test_labels[0], np.round(pred_ans[0]))
    accuracy_like = accuracy_score(test_labels[1], np.round(pred_ans[1]))

    precision_finish = precision_score(test_labels[0], np.round(pred_ans[0]))
    precision_like = precision_score(test_labels[1], np.round(pred_ans[1]))

    recall_finish = recall_score(test_labels[0], np.round(pred_ans[0]))
    recall_like = recall_score(test_labels[1], np.round(pred_ans[1]))

    f1_finish = f1_score(test_labels[0], np.round(pred_ans[0]))
    f1_like = f1_score(test_labels[1], np.round(pred_ans[1]))
    print(f"ROC-AUC (fuzzy recommendation): {roc_auc_fuzzy:.4f}")
    print(f"Combined ROC-AUC (finish): {roc_auc_finish:.4f}")
    # Print evaluation results
    print(f"ROC-AUC (finish): {roc_auc_finish:.4f}")
    print(f"ROC-AUC (like): {roc_auc_like:.4f}")

    print(f"Accuracy (finish): {accuracy_finish:.4f}")
    print(f"Accuracy (like): {accuracy_like:.4f}")

    print(f"Precision (finish): {precision_finish:.4f}")
    print(f"Precision (like): {precision_like:.4f}")

    print(f"Recall (finish): {recall_finish:.4f}")
    print(f"Recall (like): {recall_like:.4f}")

    print(f"F1 Score (finish): {f1_finish:.4f}")
    print(f"F1 Score (like): {f1_like:.4f}")