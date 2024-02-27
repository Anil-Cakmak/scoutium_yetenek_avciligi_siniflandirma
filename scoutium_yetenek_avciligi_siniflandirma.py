from helper import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from lightgbm import LGBMClassifier
import joblib

pd.set_option("display.width", 500)
pd.set_option("display.max_columns", None)

###################################################
# KEŞİFÇİ VERİ ANALİZİ VE VERİ SETİNİN HAZIRLANMASI
####################################################

attributes = pd.read_csv("datasets/scoutium_attributes.csv", sep=";")
potential_labels = pd.read_csv("datasets/scoutium_potential_labels.csv", sep=";")
df = pd.merge(attributes, potential_labels, on=["task_response_id", "match_id", "evaluator_id", "player_id"])

# Oluşturacağımız model_df veri setinde kaleci özellikleri için diğer oyuncular ve kaleci olmayan oyuncu özellikleri
# için de kaleciler eksik değerler alacak. Bunca eksik değere karşılık oluşturacağımız model_df içerisinde çok az sayıda
# satır kalecileri ifade edecek. Modelin performansı açısından sadece kaleci olmayan oyuncuların sınıflarını tahmin
# etmek üzere kaleci verilerini veri setinden çıkarıyoruz.
df = df[df["position_id"] != 1]

model_df = pd.pivot_table(df, index=["player_id", "position_id", "potential_label"],
                          columns="attribute_id", values="attribute_value")

model_df.reset_index(inplace=True)
model_df.columns = model_df.columns.astype(str)
check_df(model_df)

cat_cols = ["position_id", "potential_label"]
num_cols = [cols for cols in model_df.columns if cols.isdigit()]

for col in cat_cols:
    cat_summary(model_df, col)

# Veri setinde "below_average" sınıfına ait gözlem birimi sayısı sadece 4 olduğundan bu sınıfı average sınıfıyla bir-
# leştirerek "not highlighted" sınıfını oluşturuyoruz ve label encoding uyguluyoruz.
model_df["potential_label"] = model_df["potential_label"].apply(lambda x: 1 if x == "highlighted" else 0)

target_summary_with_cat(model_df, target="potential_label", categorical_col="position_id")

for col in num_cols:
    num_summary(model_df, col, plot=True)

for col in num_cols:
    target_summary_with_num(model_df, target="potential_label", numerical_col=col)

for col in num_cols:
    print(f"{col}", check_outlier(model_df, col))

corr = model_df[num_cols].corr().unstack().drop_duplicates().sort_values(ascending=False)
print(corr[corr > 0.9])

ss = StandardScaler()
model_df[num_cols] = ss.fit_transform(model_df[num_cols])

model_df = pd.get_dummies(model_df, columns=["position_id"], drop_first=True)
check_df(model_df)

###########
# MODEL
###########
X = model_df.drop(["potential_label", "player_id"], axis=1)
y = model_df["potential_label"]

cv_results = cross_validate(LGBMClassifier(verbose=-1, force_col_wise=True), X, y, cv=3,
                            scoring=["roc_auc", "f1", "precision", "recall", "accuracy"])

for key in cv_results.keys():
    if key.startswith("test"):
        print(f"{key}:", cv_results[key].mean())

# test_roc_auc: 0.8811450238991695
# test_f1: 0.6578147062018029
# test_precision: 0.7494074552898082
# test_recall: 0.5886939571150097
# test_accuracy: 0.8763736263736264

lgbm = LGBMClassifier(verbose=-1, force_col_wise=True)
lgbm_params = {"learning_rate": np.arange(0.01, 0.21, 0.01),
               "num_leaves": np.random.randint(5, 50, 1),
               "n_estimators": [int(x) for x in np.linspace(start=200, stop=5000, num=10)],
               "colsample_bytree": np.arange(0.1, 1.05, 0.1),
               "subsample": np.arange(0.1, 1.05, 0.1),
               "max_depth": np.arange(3, 10, 1)}

lgbm_best = RandomizedSearchCV(estimator=lgbm,
                               param_distributions=lgbm_params,
                               n_iter=150,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

lgbm_best.fit(X, y)

print(lgbm_best.best_params_)
# {'subsample': 0.7000000000000001, 'num_leaves': 19, 'n_estimators': 1266,
# 'max_depth': 8, 'learning_rate': 0.14, 'colsample_bytree': 0.9}

best_model = lgbm.set_params(**lgbm_best.best_params_)
rand_cv_results = cross_validate(best_model, X, y, cv=3, scoring=["roc_auc", "f1", "precision", "recall", "accuracy"])

for key in rand_cv_results.keys():
    if key.startswith("test"):
        print(f"{key}:", rand_cv_results[key].mean())

# test_roc_auc: 0.8775134182488183
# test_f1: 0.6448773448773449
# test_precision: 0.7880952380952381
# test_recall: 0.5526315789473685
# test_accuracy: 0.876413441630833

# Hiperparametre optimizasyonu.
lgbm = LGBMClassifier(verbose=-1, force_col_wise=True)
lgbm_grid_params = {'subsample': [0.6, 0.7, 0.8],
                    'num_leaves': [15, 19, 23],
                    'n_estimators': [1000, 1266, 1500],
                    'max_depth': [7, 8],
                    'learning_rate': [0.14, 0.1, 0.2],
                    'colsample_bytree': [0.8, 0.9, 1]}

gs_best = GridSearchCV(lgbm, lgbm_grid_params, cv=3, n_jobs=-1, verbose=True).fit(X, y)

best_model = lgbm.set_params(**gs_best.best_params_)
print(gs_best.best_params_)

# {'colsample_bytree': 0.9, 'learning_rate': 0.14, 'max_depth': 8,
# # 'n_estimators': 1000, 'num_leaves': 15, 'subsample': 0.6}

gs_cv_results = cross_validate(best_model, X, y, cv=3, scoring=["roc_auc", "f1", "precision", "recall", "accuracy"])

for key in rand_cv_results.keys():
    if key.startswith("test"):
        print(f"{key}:", rand_cv_results[key].mean())

# test_roc_auc: 0.8785014286095757
# test_f1: 0.6448773448773449
# test_precision: 0.7880952380952381
# test_recall: 0.5526315789473685
# test_accuracy: 0.876413441630833

# Final modelin kaydedilmesi.
joblib.dump(best_model, "scoutium_model.pkl")