import argparse
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from data_prep import load_emnist, make_flat_features

def main(args):
    os.makedirs(args.model_dir, exist_ok=True)
    print('Loading EMNIST...')
    ds_train, ds_test = load_emnist(split=args.split, root=args.data_dir)
    X_train, y_train = make_flat_features(ds_train)
    X_test, y_test = make_flat_features(ds_test)

    steps = [
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=args.pca)),
        ('svm', SVC(kernel=args.kernel, C=args.C, probability=True))
    ]
    pipe = Pipeline(steps)

    if args.grid_search:
        param_grid = {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 0.001, 0.01]
        }
        gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=2)
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print('Best params:', gs.best_params_)
    else:
        pipe.fit(X_train, y_train)
        model = pipe

    model_path = os.path.join(args.model_dir, args.model_name)
    joblib.dump(model, model_path)
    print('Model saved to', model_path)

    if args.eval:
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test)
        print('Test accuracy:', accuracy_score(y_test, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--model_name', default='svm_emnist.joblib')
    parser.add_argument('--pca', type=int, default=150)
    parser.add_argument('--kernel', default='rbf')
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--grid_search', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--split', default='balanced')
    args = parser.parse_args()
    main(args)
