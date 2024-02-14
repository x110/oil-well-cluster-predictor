
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df_train):
    y0 = df_train['cluster']
    X = df_train.drop(columns={'cluster', 'well'})
    return X, y0


def encode_labels(y0):
    class_labels = ['constant', 'multi', 'normal', 'rapid']
    label_encoder = LabelEncoder()
    label_encoder.fit(class_labels)
    y = label_encoder.transform(y0)
    return y