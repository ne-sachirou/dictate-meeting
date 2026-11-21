Whisper (kotoba-tech/kotoba-whisper-v2.2) で日本語音聲を文字起こしし、2人分の話者ラベル附きでテキストファイルに保存します。

呼び出し方:

```sh
#!/bin/bash
set -eux

ffmpeg -i ./*.mp4 \
       -ac 1 \
       -af "loudnorm" \
       -ar 16000 \
       meeting.wav

set +x
#云々="$(echo -n '云々' | openssl enc -e -aes-256-cbc -pbkdf2 -base64 -pass 'pass:云々')"
HF_TOKEN="$(echo "$云々" | openssl enc -d -aes-256-cbc -pbkdf2 -base64 -pass 'pass:云々')"
set -x
export HF_TOKEN
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@9/lib:${DYLD_LIBRARY_PATH:-}"
(cd dictate && pipenv run python dictate.py ../meeting.wav)
```
