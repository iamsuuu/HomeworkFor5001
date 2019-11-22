import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle
import xgboost as xgb


def get_data():
    # 读入数据
    df_train = pd.read_csv('/Users/data/train.csv')
    df_test = pd.read_csv('/Users/data/test.csv')
    # save id & drop id
    train_ID = df_train['id']
    test_ID = df_test['id']
    df_train.drop("id", axis=1, inplace=True)
    df_test.drop("id", axis=1, inplace=True)
    # 分离特征与预测值
    train_label = df_train['playtime_forever']

    # 删除两个很大的price数据
    df_train = df_train.drop(df_train[df_train['price'] > 30800].index)

    # 特征处理前准备数据
    train_feature = df_train.copy()
    train_feature = train_feature.drop('playtime_forever', axis=1)
    test_feature = df_test.copy()
    data = pd.concat([train_feature, test_feature], axis=0)
    ntrain = df_train.shape[0]
    ntest = df_test.shape[0]
    y_train = df_train.playtime_forever.values
    print("\nThe train data size after dropping Id feature and price outliers is : {} ".format(train_feature.shape))
    print("The test data size after dropping Id feature is : {} ".format(test_feature.shape))
    print("all data size is : {}".format(data.shape))
    # 重新设置index
    data = data.reset_index(drop=True)

    return data, test_ID, y_train, ntrain, ntest


def preprocessing(data):
    # 字符串转换为时间格式
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])
    data['release_date'] = pd.to_datetime(data['release_date'])
    # 将发布游戏与购买游戏之间的时间差转换为天数
    data['date_interval'] = data['purchase_date'] - data['release_date']
    data['date_interval'].apply(str)
    data['duriation_days'] = data['date_interval'] / np.timedelta64(1, 'D')
    # true = 1, false = 0， is_free 列将类别转换为数据
    data['is_free'] = data['is_free'].replace([False, True], [0, 1])

    # 对string数据进行处理，str.get_dummies(' ')
    genres = data['genres'].str.get_dummies(",")
    categories = data['categories'].str.get_dummies(",")
    tags = data['tags'].str.get_dummies(",")
    # 使用pca
    pca = PCA(n_components=0.9)
    genres = pca.fit_transform(genres)
    categories = pca.fit_transform(categories)
    tags = pca.fit_transform(tags)
    # numpy转换成dataframe
    genres = pd.DataFrame(genres)
    categories = pd.DataFrame(categories)
    tags = pd.DataFrame(tags)

    genres = genres.rename(columns=lambda x: 'genres_' + str(x))
    categories = categories.rename(columns=lambda x: 'categories_' + str(x))
    tags = tags.rename(columns=lambda x: 'tags_' + str(x))

    # 删除多余列
    droped = ['genres', 'categories', 'tags', 'purchase_date', 'release_date', 'date_interval']
    feature_data = data.drop(droped, axis=1)

    feature_data = pd.concat([feature_data, genres, categories, tags], axis=1)

    print("The data size after feature engineering is : {} ".format(feature_data.shape))
    return feature_data


def fill_na(feature_data):
    for col in ('total_positive_reviews','total_negative_reviews','duriation_days'):
        feature_data[col]=feature_data[col].fillna(0)
    return feature_data


def split(feature_data,ntrain):
    X_train = feature_data[0:ntrain]
    X_test = feature_data[ntrain:]
    return X_train,X_test


def train(X_train,y_train):
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                 learning_rate=0.05, max_depth=3,
                                 min_child_weight=1.7817, n_estimators=49,
                                 reg_alpha=0.4640, reg_lambda=0.8571,
                                 subsample=0.5213, silent=1,
                                 random_state=7, nthread=-1)
    X1, _,Y1,_= train_test_split(X_train, y_train, test_size=0.1)
    model_xgb.fit(X1,Y1)
    # Save to file in the current working directory
    pkl_filename = "model_xgb.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model_xgb, file)
    print("model_xgb saved")

def predict(X_test):
    pred = pickle.load(open("model_xgb.pkl", 'rb'))
    vec = pred.predict(X_test)
    playtime = np.array(vec)
    return playtime



data,test_ID,y_train,ntrain,ntest = get_data()
feature_data = preprocessing(data)
feature_data = fill_na(feature_data)
X_train,X_test= split(feature_data,ntrain)
train(X_train,y_train)

print("Getting playtime_forever...")
ensemble=predict(X_test)

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['playtime_forever'] = ensemble
sub.to_csv('submission.csv',index=False)

