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
#F05452E2822D="$(echo -n '云々' | openssl enc -e -aes-256-cbc -pbkdf2 -base64 -pass 'pass:c00e7cdf-c137-4f4a-9ded-35824c740814')"
HF_TOKEN="$(echo "$F05452E2822D" | openssl enc -d -aes-256-cbc -pbkdf2 -base64 -pass 'pass:c00e7cdf-c137-4f4a-9ded-35824c740814')"
set -x
export HF_TOKEN
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@9/lib:${DYLD_LIBRARY_PATH:-}"
(cd dictate && pipenv run python dictate.py ../meeting.wav)
```
