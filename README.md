# initial-research-verification
研究の初期検証

## TensorBoardの開き方

```bash
ssh -L 6006:localhost:6006 ishii@192.168.34.32

cd _iceberg/speaker_direction_estimation

source .venv/bin/activate

tensorboard --logdir runs --port 6006
```

ブラウザで以下を開く

http://localhost:6006


## lsコマンドの色表示設定

### .bashrc に追加

```bash
alias ls='ls --color=auto'
PS1='\[\e[32m\]\u@\h\[\e[0m\]:\[\e[34m\]\w\[\e[0m\]\$ '
export LS_COLORS='di=1;36'
```

### 設定を反映

```bash
source ~/.bashrc
```

### .bash_profile を作成（ログインシェル用）

```bash
cat > ~/.bash_profile <<'EOF'
if [ -f ~/.bashrc ]; then
    . ~/.bashrc
fi
EOF
```

```bash
source ~/.bash_profile
```

### 確認

```bash
alias ls
```

## tmux（ターミナルセッション管理）

### 目的

リモートサーバーで長時間の処理（学習など）を実行する際、  
SSH接続が切れても処理が止まらないようにするために tmux を使用する。

### インストール

```bash
sudo apt update
sudo apt install tmux
```

### 新しいセッションを開始

```bash
tmux
```

### セッションから離れる（処理は継続）

```
Ctrl + b → d
```

### セッション一覧

```bash
tmux ls
```

### セッションに再接続

```bash
tmux a -t 0
```

### セッション終了

```bash
exit
```

## train.py の使い方

`train.py` は以下の 3 つのモードを持つ。

- `precompute` : 音声データから特徴量を事前計算して保存する
- `train` : precompute 済みデータを使って学習する
- `evaluate` : 学習済みモデルで評価を行う

## 1. precompute(事前処理)

### 目的

音声データのメルスペクトログラムを事前に計算し、`.npy` と `metadata.csv` として保存する。  
学習時に毎回特徴量を計算しないため、学習を高速に進められる。

### 実行

```bash
python train.py precompute
```

### オプション

--root   元の音声データのディレクトリ（デフォルト: ./train_dataset）
--out    precompute 結果の保存先（デフォルト: ./precomputed/train）

### 実行例

```
python train.py precompute \
  --root ./train_dataset \
  --out ./precomputed/train
```
