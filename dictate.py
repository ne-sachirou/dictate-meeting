import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from pyannote.audio import Pipeline
from transformers import pipeline


@dataclass
class Segment:
    start: float
    end: float
    speaker: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Whisper (kotoba-tech/kotoba-whisper-v2.2) で日本語音声を文字起こしし、"
            "2人分の話者ラベル付きでテキストファイルに保存します。"
        )
    )
    parser.add_argument("wav_path", type=Path, help="入力 WAV ファイルのパス")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力テキストファイルのパス（省略時は入力と同名で .txt）",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=30,
        help="長時間音声の分割秒数（デフォルト: 30）",
    )
    parser.add_argument(
        "--stride-seconds",
        type=int,
        default=5,
        help="チャンク間の重なり秒数（デフォルト: 5）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="推論バッチサイズ（デフォルト: 8）",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face のアクセストークン（未指定なら環境変数 HF_TOKEN から取得）",
    )
    return parser


def resolve_output_path(wav_path: Path, output: Path | None) -> Path:
    if output is not None:
        return output
    return wav_path.with_suffix(".txt")


def require_hf_token(cli_token: str | None) -> str:
    token = cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "Hugging Face のアクセストークンが必要です。"
            " --hf-token か環境変数 HF_TOKEN を設定してください。"
        )
    return token


def load_audio(wav_path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(str(wav_path))
    return waveform, sample_rate


def diarize(
    waveform: torch.Tensor,
    sample_rate: int,
    hf_token: str,
) -> list[Segment]:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    if torch.cuda.is_available():
        diarization_pipeline.to(torch.device("cuda"))

    diarization_result = diarization_pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=2, # NOTE: 會議には常に2人が參加してゐるを假定してゐる
    )
    diarization = getattr(diarization_result, "speaker_diarization", diarization_result)

    segments: list[Segment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(Segment(start=turn.start, end=turn.end, speaker=speaker))

    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def format_timestamp(seconds: float) -> str:
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{sec:05.2f}"


def slice_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    start: float,
    end: float,
) -> torch.Tensor:
    start_sample = int(start * sample_rate)
    end_sample = max(int(end * sample_rate), start_sample + 1)
    segment = waveform[:, start_sample:end_sample]
    if segment.size(0) > 1:
        segment = segment.mean(dim=0, keepdim=True)
    return segment


def build_asr_pipeline() -> tuple:
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr = pipeline(
        "automatic-speech-recognition",
        model="kotoba-tech/kotoba-whisper-v2.2",
        device=device,
        torch_dtype=torch_dtype,
    )
    return asr, device


def transcribe_segment(
    asr,
    audio_segment: torch.Tensor,
    sample_rate: int,
    chunk_seconds: int,
    stride_seconds: int,
    batch_size: int,
) -> str:
    audio_array = audio_segment.squeeze(0).cpu().numpy()
    result = asr(
        {"array": audio_array, "sampling_rate": sample_rate},
        chunk_length_s=chunk_seconds,
        stride_length_s=(stride_seconds, stride_seconds),
        batch_size=batch_size,
        generate_kwargs={"language": "ja", "task": "transcribe"},
    )
    return result.get("text", "").strip()


def map_speakers(segments: list[Segment]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    next_id = 1
    for seg in segments:
        if seg.speaker not in mapping:
            mapping[seg.speaker] = f"SPEAKER_{next_id}"
            next_id += 1
        if next_id > 2:
            break
    return mapping


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    wav_path = args.wav_path
    if not wav_path.exists():
        raise SystemExit(f"入力ファイルが見つかりません: {wav_path}")

    output_path = resolve_output_path(wav_path, args.output)
    hf_token = require_hf_token(args.hf_token)

    waveform, sample_rate = load_audio(wav_path)
    segments = diarize(waveform, sample_rate, hf_token)
    speaker_map = map_speakers(segments)
    asr, _ = build_asr_pipeline()

    lines: list[str] = []
    for segment in segments:
        audio_segment = slice_waveform(waveform, sample_rate, segment.start, segment.end)
        text = transcribe_segment(
            asr,
            audio_segment,
            sample_rate,
            args.chunk_seconds,
            args.stride_seconds,
            args.batch_size,
        )
        if not text:
            continue
        speaker_label = speaker_map.get(segment.speaker, segment.speaker)
        start_ts = format_timestamp(segment.start)
        end_ts = format_timestamp(segment.end)
        lines.append(f"[{start_ts} - {end_ts}] {speaker_label}: {text}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"書き出し完了: {output_path}")


if __name__ == "__main__":
    main()
