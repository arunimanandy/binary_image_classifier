import argparse, json, time
from pathlib import Path
import yaml, mlflow, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from .model import build_model, CLASS_NAMES


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def loaders(processed_dir, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    test_tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    dsets = {s: datasets.ImageFolder(str(Path(processed_dir)/s), transform=train_tf if s=='train' else test_tf) for s in ['train','val','test']}
    return {s: DataLoader(dsets[s], batch_size=batch_size, shuffle=(s=='train'), num_workers=num_workers) for s in dsets}, dsets


def evaluate(model, loader):
    y_true, y_pred = [], []
    loss_fn = nn.CrossEntropyLoss(); losses=[]
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            out = model(x); losses.append(loss_fn(out,y).item())
            y_true.extend(y.tolist()); y_pred.extend(out.argmax(1).tolist())
    return sum(losses)/max(1,len(losses)), accuracy_score(y_true,y_pred), y_true, y_pred


def plot_loss(train_losses, val_losses, out):
    plt.figure(); plt.plot(train_losses, label='train_loss'); plt.plot(val_losses, label='val_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.tight_layout(); plt.savefig(out); plt.close()


def plot_confusion(y_true, y_pred, out):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    plt.figure(); plt.imshow(cm); plt.xticks([0,1], CLASS_NAMES); plt.yticks([0,1], CLASS_NAMES)
    for i in range(2):
        for j in range(2): plt.text(j,i,str(cm[i,j]),ha='center',va='center')
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.tight_layout(); plt.savefig(out); plt.close()


def main(config_path):
    cfg = load_config(config_path)
    Path('models').mkdir(exist_ok=True); Path('reports').mkdir(exist_ok=True)
    mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri']); mlflow.set_experiment(cfg['mlflow']['experiment_name'])
    model = build_model(); opt = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate']); loss_fn = nn.CrossEntropyLoss()
    dl, _ = loaders(cfg['data']['processed_dir'], cfg['training']['batch_size'], cfg['training']['num_workers'])
    train_losses, val_losses = [], []
    with mlflow.start_run(run_name=f'cnn-{int(time.time())}'):
        mlflow.log_params({'epochs': cfg['training']['epochs'], 'batch_size': cfg['training']['batch_size'], 'lr': cfg['training']['learning_rate'], 'image_size': cfg['image_size']})
        for epoch in range(cfg['training']['epochs']):
            model.train(); running=[]
            for x,y in dl['train']:
                opt.zero_grad(); out=model(x); loss=loss_fn(out,y); loss.backward(); opt.step(); running.append(loss.item())
            train_loss = sum(running)/max(1,len(running)); val_loss, val_acc, _, _ = evaluate(model, dl['val'])
            train_losses.append(train_loss); val_losses.append(val_loss)
            mlflow.log_metric('train_loss', train_loss, step=epoch); mlflow.log_metric('val_loss', val_loss, step=epoch); mlflow.log_metric('val_accuracy', val_acc, step=epoch)
        test_loss, test_acc, yt, yp = evaluate(model, dl['test'])
        plot_loss(train_losses, val_losses, 'reports/loss_curve.png'); plot_confusion(yt, yp, 'reports/confusion_matrix.png')
        torch.save({'model_state_dict': model.state_dict(), 'class_names': CLASS_NAMES, 'image_size': cfg['image_size']}, cfg['model']['artifact_path'])
        metrics={'test_loss': test_loss, 'test_accuracy': test_acc}
        Path('reports/metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        mlflow.log_metrics(metrics); mlflow.log_artifact(cfg['model']['artifact_path']); mlflow.log_artifacts('reports')
        print(metrics)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config', default='configs.yaml'); args=parser.parse_args(); main(args.config)
