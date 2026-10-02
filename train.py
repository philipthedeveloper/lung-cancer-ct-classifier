"""Train a transfer-learning classifier (EfficientNetB0) on the chest CT-scan dataset.

Usage:
    python train.py                    # default: 20 head epochs + 30 fine-tune epochs
    python train.py --epochs 10 --fine-tune-epochs 10

Outputs:
    models/model.keras            trained model (takes raw 0-255 RGB 224x224 images)
    models/class_names.json       class index -> name mapping used by the GUI
    results/metrics.json          test accuracy, macro F1, per-class precision/recall/F1
    results/classification_report.txt
    results/confusion_matrix.png
    results/training_curves.png
    results/gradcam.png           Grad-CAM heatmaps for sample test images
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# The train/valid folders carry staging suffixes (e.g. "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib")
# while test folders don't, so classes are matched on the part before the first underscore.
DISPLAY_NAMES = {
    'adenocarcinoma': 'Adenocarcinoma',
    'large.cell.carcinoma': 'Large Cell Carcinoma',
    'normal': 'Normal',
    'squamous.cell.carcinoma': 'Squamous Cell Carcinoma',
}


def load_split(directory, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory, label_mode='categorical', image_size=IMG_SIZE,
        batch_size=BATCH_SIZE, shuffle=shuffle, seed=SEED)
    names = [n.split('_')[0] for n in ds.class_names]
    return ds, names


def load_data(data_dir):
    train_ds, train_names = load_split(data_dir / 'train', shuffle=True)
    # shuffle=False keeps predictions aligned with labels when computing metrics
    valid_ds, valid_names = load_split(data_dir / 'valid', shuffle=False)
    test_ds, test_names = load_split(data_dir / 'test', shuffle=False)
    if not train_names == valid_names == test_names:
        raise ValueError(f'Class folders differ between splits: {train_names}, {valid_names}, {test_names}')
    return train_ds, valid_ds, test_ds, train_names


def build_model(num_classes):
    # EfficientNet includes its own rescaling/normalisation, so it expects raw 0-255 pixels
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='imagenet', input_shape=IMG_SIZE + (3,))
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(base.output)
    x = tf.keras.layers.Dropout(0.3, name='head_dropout')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    return tf.keras.Model(base.input, outputs), base


def unfreeze_top(base, n_layers):
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    # BatchNorm stays frozen so the small batches don't wreck the pretrained statistics
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def augment(train_ds):
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.1),
    ])
    return train_ds.map(lambda x, y: (augmentation(x, training=True), y),
                        num_parallel_calls=tf.data.AUTOTUNE)


def class_weights(train_ds):
    labels = np.concatenate([np.argmax(y, axis=1) for _, y in train_ds])
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))


def fit(model, train_ds, valid_ds, epochs, lr, weights):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    ]
    return model.fit(train_ds, validation_data=valid_ds, epochs=epochs,
                     class_weight=weights, callbacks=callbacks)


def predict(model, ds):
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in ds])
    y_prob = model.predict(ds)
    return y_true, np.argmax(y_prob, axis=1)


def plot_curves(histories, path):
    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history['accuracy']
        val_acc += h.history['val_accuracy']
        loss += h.history['loss']
        val_loss += h.history['val_loss']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (train, val, title) in zip(axes, [(acc, val_acc, 'Accuracy'), (loss, val_loss, 'Loss')]):
        ax.plot(train, label='Train')
        ax.plot(val, label='Validation')
        ax.axvline(len(histories[0].history['loss']) - 0.5, color='grey', linestyle='--', label='Fine-tuning starts')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, labels, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Test set confusion matrix')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def gradcam(model, image, class_idx, conv_layer='top_activation'):
    grad_model = tf.keras.Model(model.input, [model.get_layer(conv_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(image[np.newaxis])
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return tf.image.resize(cam[..., np.newaxis], IMG_SIZE).numpy()[..., 0]


def plot_gradcam(model, test_ds, labels, path, per_class=2):
    images, targets = [], []
    for x, y in test_ds.unbatch():
        cls = int(np.argmax(y))
        if targets.count(cls) < per_class:
            images.append(x.numpy())
            targets.append(cls)
        if len(targets) == per_class * len(labels):
            break
    fig, axes = plt.subplots(per_class, len(labels), figsize=(4 * len(labels), 4 * per_class))
    order = np.argsort(targets, kind='stable')
    for i, idx in enumerate(order):
        ax = axes[i % per_class, i // per_class]
        img = images[idx]
        pred = int(np.argmax(model.predict(img[np.newaxis], verbose=0)))
        ax.imshow(img.astype('uint8'))
        ax.imshow(gradcam(model, img, pred), cmap='jet', alpha=0.4)
        ax.set_title(f'True: {labels[targets[idx]]}\nPred: {labels[pred]}', fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', type=Path, default=ROOT)
    parser.add_argument('--epochs', type=int, default=20, help='epochs with the backbone frozen')
    parser.add_argument('--fine-tune-epochs', type=int, default=30, help='epochs fine-tuning the top layers')
    parser.add_argument('--unfreeze', type=int, default=100, help='number of backbone layers to unfreeze')
    parser.add_argument('--fine-tune-lr', type=float, default=1e-4, help='learning rate for fine-tuning')
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    models_dir, results_dir = ROOT / 'models', ROOT / 'results'
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    train_ds, valid_ds, test_ds, class_keys = load_data(args.data_dir)
    labels = [DISPLAY_NAMES.get(k, k) for k in class_keys]
    weights = class_weights(train_ds)
    print(f'Classes: {labels}\nClass weights: {weights}')

    train_aug = augment(train_ds).prefetch(tf.data.AUTOTUNE)
    valid_ds = valid_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    model, base = build_model(len(labels))
    print('\n=== Phase 1: training classification head (backbone frozen) ===')
    h1 = fit(model, train_aug, valid_ds, args.epochs, 1e-3, weights)

    print(f'\n=== Phase 2: fine-tuning top {args.unfreeze} backbone layers ===')
    unfreeze_top(base, args.unfreeze)
    h2 = fit(model, train_aug, valid_ds, args.fine_tune_epochs, args.fine_tune_lr, weights)

    model.save(models_dir / 'model.keras')
    (models_dir / 'class_names.json').write_text(json.dumps(labels, indent=2))

    y_true, y_pred = predict(model, test_ds)
    val_true, val_pred = predict(model, valid_ds)
    report = classification_report(y_true, y_pred, target_names=labels, digits=3)
    report_dict = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    metrics = {
        'model': 'EfficientNetB0 (ImageNet) + fine-tuning',
        'train_images': int(sum(1 for _ in train_ds.unbatch())),
        'test_images': int(len(y_true)),
        'validation_accuracy': round(accuracy_score(val_true, val_pred), 4),
        'test_accuracy': round(accuracy_score(y_true, y_pred), 4),
        'test_macro_f1': round(f1_score(y_true, y_pred, average='macro'), 4),
        'test_weighted_f1': round(f1_score(y_true, y_pred, average='weighted'), 4),
        'per_class': {
            name: {k: round(report_dict[name][k], 4) for k in ('precision', 'recall', 'f1-score', 'support')}
            for name in labels
        },
    }
    (results_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    (results_dir / 'classification_report.txt').write_text(report)

    plot_curves([h1, h2], results_dir / 'training_curves.png')
    plot_confusion(confusion_matrix(y_true, y_pred), labels, results_dir / 'confusion_matrix.png')
    plot_gradcam(model, test_ds, labels, results_dir / 'gradcam.png')

    print('\n=== Test set results ===')
    print(report)
    print(f"Test accuracy: {metrics['test_accuracy']:.2%} | Macro F1: {metrics['test_macro_f1']:.3f}")
    print(f'Saved model to {models_dir} and results to {results_dir}')


if __name__ == '__main__':
    main()
