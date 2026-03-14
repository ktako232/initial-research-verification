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
