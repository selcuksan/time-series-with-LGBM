import warnings
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

df_ = pd.read_csv("../input/5g-smart-prediction/NR_CELL_History_PFM_train.csv")
df = df_.copy()

test = df.loc[(df["Time"] >= "2021-09-20") & (df["Time"] <= "2021-10-17"), :]

df_ = pd.read_csv("../input/5g-smart-prediction/NR_CELL_History_PFM_train.csv")
df_desc = pd.read_csv("../input/5g-smart-prediction/Data_Description.csv")
df = df_.copy()

test = df.loc[(df["Time"] >= "2021-09-20") & (df["Time"] <= "2021-10-17"), :]


#####################################################
# Exploratory Data Analysis
#####################################################

def check_df(dataframe, head=5):
    """
    Veri setindekinin genel özelliklerini ve istatistiklerini verir.
    """
    print("########### Shape ###########")
    print(dataframe.shape)
    print("########### Types  ###########")
    print(dataframe.dtypes)
    print("########### Head ###########")
    print(dataframe.head(head))
    print("########### Tail ###########")
    print(dataframe.tail(head))
    print("########### Null values ###########")
    print(dataframe.isnull().sum())
    print("########### Describe ###########")
    print(dataframe.describe().T)
    print("########### ###########")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def missing_vs_target(dataframe, target, na_columns):
    """
    Veri setindeki eksik değerlerin hedef değişkenle ne kadar ilişkili oldugunu gösterir.
    """
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
        na_flag_df = temp_df.loc[:, temp_df.columns.str.contains("_NA_")]
    for col in na_flag_df:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby([col])[target].mean(),
                            "COUNT": temp_df.groupby([col])[target].count()}), end="\n\n\n")


def missing_values_table(dataframe, na_name=True):
    """
    Veri setinde eksik değer bulunduran değişkenleri ve eksik değerlerin tüm veri setine oranını verir.
    """
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum()
    ratio = (dataframe[na_columns].isnull().sum().sort_values(ascending=False) / dataframe.shape[0]).sort_values(
        ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)
    if na_name:
        return na_columns


def check_outlier(dataframe, col_name):
    """
    Veri setindeki değişkenlerin aykırı değerlere sahip olup olmadıgını söyler.
    """
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    return False


def grab_outliers(dataframe, col_name, index=False, plot=False):
    """
    Veri setindeki aykırı değerlerin index bilgilerini getirir.
    """
    import seaborn as sns
    from matplotlib import pyplot as plt
    if plot:
        sns.boxplot(dataframe[col_name])
        plt.show()
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head(5))
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


#####################################################
# FEATURE ENGINEERING
#####################################################

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    """
    Veri setindeki aykırı değerlerini baskılamak için gerekli alt ve üst limiti belirler.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    """
    Veri setindeki aykırı değerlerini alt ve üst limite göre baskılar.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def outlier_values(dataframe):
    """
    Veri setindeki aykırı değerleri alt ve üst limite göre baskılar.Check ve baskılama işlemlerini birlikte yapan fonksiyondur.
    """
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    for col in num_cols:
        print(col, check_outlier(dataframe, col), sep=":")
    for col in num_cols:
        print(col, grab_outliers(dataframe, col, index=True, plot=False).shape[0] / dataframe.shape[0], sep=":")
    num_features = [col for col in num_cols if col not in ["N.ThpVol.DL", "N.User.RRCConn.Avg"]]
    for col in num_features:
        replace_with_thresholds(dataframe, col)


def missing_values(dataframe):
    """
    Veri setindeki eksik değer işlemlerini yapar. Eksik değerleri kendinden 1 önceki değerle doldurur
    """
    na_columns = missing_values_table(dataframe)
    missing_vs_target(dataframe, "N.ThpVol.DL", na_columns)
    missing_vs_target(dataframe, "N.User.RRCConn.Avg", na_columns)
    dataframe = dataframe.fillna(dataframe.bfill())
    return dataframe


#####################################################
# Feature Extraction
#####################################################
def lag_features(dataframe, lags, target):
    """
    lag/shift değişkenleri oluşturur.
    """
    for lag in lags:
        dataframe[target + '_lag_' + str(lag)] = dataframe.groupby(["Site Name", "Cell Name"])[target].transform(
            lambda x: x.shift(lag))
    return dataframe


def create_date_features(dataframe):
    """
    Zamana bağlı yeni değişkenler oluşturur.
    """
    dataframe["Time"] = pd.to_datetime(dataframe["Time"])
    dataframe['month'] = dataframe.Time.dt.month
    dataframe['day_of_month'] = dataframe.Time.dt.day
    dataframe['day_of_year'] = dataframe.Time.dt.dayofyear
    dataframe['week_of_year'] = dataframe.Time.dt.weekofyear
    dataframe['day_of_week'] = dataframe.Time.dt.dayofweek
    dataframe['year'] = dataframe.Time.dt.year
    dataframe["is_wknd"] = dataframe.Time.dt.weekday // 5
    dataframe['is_month_start'] = dataframe.Time.dt.is_month_start.astype(int)
    dataframe['is_month_end'] = dataframe.Time.dt.is_month_end.astype(int)
    return dataframe


#############################
# Data Preprocessing
#############################
def label_encoding(dataframe, column):
    """
    Kategorik değişkenlere Label Encoding işlemi uygular.
    """
    l_encoder = LabelEncoder().fit(dataframe[column])
    encoded_col = l_encoder.transform(dataframe[column])
    dataframe[column + "_encoded"] = encoded_col
    columns = [col for col in dataframe.columns if col not in ["Cell Name"]]
    dataframe = dataframe[columns]
    return dataframe, l_encoder


def log_transform(dataframe, target):
    """
    Bağımlı değişkene logaritmik dönüşüm uygular
    """
    dataframe[target] = np.log1p(dataframe[target].values)
    return dataframe


#############################
# MODEL EVALUATING FUNCS
#############################
# SMAPE
def smape(preds, target):
    """
    sMAPE performans metriğini hesaplar.
    """
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


#############################
# MODEL TRAINING
#############################

def get_feature_cols(dataframe, dependent_variable):
    """
    İlgili bağımlı değişken dışarıda bırakılır ve geri kalan değişkenler bağımsız değişken olarak geri döndürülür.
    """
    feature_cols = [col for col in dataframe.columns if
                    col not in [dependent_variable, "Time", "Cell Name"]]
    return feature_cols


def create_models(dataframe):
    """
    2 bağımlı değişken için ayrı ayrı model geliştirilir
    """
    models = []
    dependent_variables = ["N.User.RRCConn.Avg", "N.ThpVol.DL"]
    for target in dependent_variables:
        df_ = log_transform(dataframe, target)
        feature_cols = get_feature_cols(df_, target)
        model = train_model(df_, feature_cols, target)
        joblib.dump(model, target + "_model.pkl")  # Model kayıt edilir.
        models.append(model)
    return models


def train_model(dataframe, feature_cols, target_variable):
    """
    Tüm veri seti ile model eğitilir. Final modeli döndürülür.
    """
    model_with_val_set = create_models_with_val_sets(
        dataframe, feature_cols, target_variable)
    Y_train = dataframe[target_variable]
    X_train = dataframe[feature_cols]

    lgb_params = {'num_leaves': 10,
                  'learning_rate': 0.02,
                  'feature_fraction': 0.8,
                  'max_depth': 5,
                  'verbose': 0,
                  'nthread': -1,
                  "num_boost_round": model_with_val_set.best_iteration}
    lgbtrain_all = lgb.Dataset(
        data=X_train, label=Y_train, feature_name=feature_cols)

    final_model = lgb.train(lgb_params, lgbtrain_all,
                            num_boost_round=model_with_val_set.best_iteration)

    return final_model


def create_models_with_val_sets(dataframe, feature_cols, target_variable):
    """
    Model geliştirilip Validation setleri üzerinden tahmin gerçekleştirilir. Model optimizasyonu için gerekli best_iteration parametresi elde edilir.
    """
    train_set = dataframe.loc[(dataframe["Time"] < "2021-08-20"), :]
    val_set = dataframe.loc[(dataframe["Time"] >= "2021-08-20")
                            & (dataframe["Time"] <= "2021-09-19"), :]
    Y_train = train_set[target_variable]
    X_train = train_set[feature_cols]

    Y_val = val_set[target_variable]
    X_val = val_set[feature_cols]

    # Parametre optimizasyonu
    lgb_params = {'num_leaves': 10,
                  'learning_rate': 0.02,
                  'feature_fraction': 0.8,
                  'max_depth': 5,
                  'verbose': 0,
                  'num_boost_round': 1000,
                  'early_stopping_rounds': 200,  # erken durdurma parametresi over fitting olmaması için belirlendi.
                  'nthread': -1}

    lgbtrain = lgb.Dataset(data=X_train, label=Y_train,
                           feature_name=feature_cols)

    lgbval = lgb.Dataset(data=X_val, label=Y_val,
                         reference=lgbtrain, feature_name=feature_cols)

    model = lgb.train(lgb_params, lgbtrain,
                      valid_sets=[lgbtrain, lgbval],
                      num_boost_round=lgb_params['num_boost_round'],
                      early_stopping_rounds=lgb_params['early_stopping_rounds'],
                      feval=lgbm_smape,
                      verbose_eval=100)

    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

    print(smape(np.expm1(y_pred_val), np.expm1(Y_val)))
    return model


#############################
# MODEL PREDICTING
#############################
def predict(NR_CELL_History_PFM_csv, cell_name, trainDf_for_encoder):
    """
    Tahminleme fonksiyonu. Output.csv dosyası ile birlikte modellerin ilgili cell name parametresindeki başarısını geri döndürür. trainDf_for_encoder parametresine label encoding yapabilmek için ihtiyaç duyulur
    """
    scores = {}
    outputs = []
    NR_CELL_History_PFM_csv = missing_values(NR_CELL_History_PFM_csv)  # Test setindeki eksik değerler temizlenir

    outlier_values(NR_CELL_History_PFM_csv)  # Test setindeki aykırı değerler baskılanır.

    NR_CELL_History_PFM_csv = create_date_features(NR_CELL_History_PFM_csv)  # Date featureları oluşturulur.

    NR_CELL_History_PFM_csv = NR_CELL_History_PFM_csv.loc[
        NR_CELL_History_PFM_csv["Cell Name"] == cell_name]  # CELL NAME parametresine göre filtrelenir.

    _, encoder = label_encoding(trainDf_for_encoder,
                                "Cell Name")  # Train setinin encoder ı elde edilir. Hata olmaması için şart.
    encoded_col = encoder.transform(
        NR_CELL_History_PFM_csv["Cell Name"])  # Test setindeki cell name değişkeni encode edilir.
    NR_CELL_History_PFM_csv["Cell Name" + "_encoded"] = encoded_col  # encoded değişken dataframe e eklenir.

    columns = [col for col in NR_CELL_History_PFM_csv.columns if col not in ["Cell Name"]]
    NR_CELL_History_PFM_csv = NR_CELL_History_PFM_csv[columns]

    dependent_variables = ["N.User.RRCConn.Avg", "N.ThpVol.DL"]
    time_index = NR_CELL_History_PFM_csv["Time"]

    for target in dependent_variables:
        df_ = log_transform(NR_CELL_History_PFM_csv, target)  # Her bir bağımlı değişken için log dönüşüm uygulanır.
        feature_cols = get_feature_cols(df_, target)  # bağımsız değişklenler getirilir.
        Y_test = df_[target]  # X ve Y belirlenir.
        X_test = df_[feature_cols]

        model = joblib.load(target + "_model.pkl")  # eğitilmiş model localden yüklenir.
        test_preds = model.predict(X_test, num_iteration=model.best_iteration)  # tahminleme yapılır.
        smape_score = smape(np.expm1(test_preds), np.expm1(Y_test))  # hata score u hesaplanır

        scores[target + "_model.pkl sMAPE Score"] = smape_score
        outputs.append(pd.DataFrame({
            "date": time_index,
            target: test_preds
        }))

    output_csv = pd.concat([outputs[0], outputs[1]], axis=1).reset_index()
    output_csv.columns = ["index", "date",
                          "N.User.RRCConn.Avg", "date2", "N.ThpVol.DL"]
    output_csv.drop(["date2", "index"], axis=1, inplace=True)
    print(f"{cell_name} için TAHMİNler YAPILDI")
    return output_csv, scores


#############################
# FEATURE IMPORTANCES
#############################
def plot_lgb_importances(model, target, plot=False, num=10):
    """
    Eğitilen modelin feature importances değerleri görselleştirilir.
    """
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title(target + " Modeli")
        plt.tight_layout()
        plt.show()
    else:
        print(feat_imp.head(num))
    return feat_imp


df = missing_values(df)
outlier_values(df)
df = create_date_features(df)
df, encoder = label_encoding(df, "Cell Name")
trained_models = create_models(df)

result = predict(test, "9783299.03.0", df_)

for target in ["N.User.RRCConn.Avg", "N.ThpVol.DL"]:
    df = log_transform(df, target)
    model = joblib.load(target + "_model.pkl")
    plot_lgb_importances(model, target, num=30, plot=True)
