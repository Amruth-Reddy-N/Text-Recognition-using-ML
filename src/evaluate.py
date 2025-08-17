import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data_prep import load_emnist, make_flat_features

def evaluate(model_path, split='balanced', data_dir='./data'):
    print('Loading model', model_path)
    model = joblib.load(model_path)
    ds_train, ds_test = load_emnist(split=split, root=data_dir)
    X_test, y_test = make_flat_features(ds_test)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print('Test accuracy:', acc)
    print('\nClassification report:\n')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, fmt='d')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/svm_emnist.joblib')
    parser.add_argument('--split', default='balanced')
    parser.add_argument('--data_dir', default='./data')
    args = parser.parse_args()
    evaluate(args.model, split=args.split, data_dir=args.data_dir)
