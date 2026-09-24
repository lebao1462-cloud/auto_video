from buzz.model_loader import ModelType
from buzz.transcriber.file_transcriber import FileTranscriber
from buzz.transcriber.transcriber import FileTranscriptionTask


def create_file_transcriber(task: FileTranscriptionTask) -> FileTranscriber:
    """Create the existing Buzz file transcriber for a task's selected model."""
    model_type = task.transcription_options.model.model_type

    if model_type == ModelType.OPEN_AI_WHISPER_API:
        from buzz.transcriber.openai_whisper_api_file_transcriber import (
            OpenAIWhisperAPIFileTranscriber,
        )

        return OpenAIWhisperAPIFileTranscriber(task=task)

    if model_type in {
        ModelType.WHISPER_CPP,
        ModelType.HUGGING_FACE,
        ModelType.WHISPER,
        ModelType.FASTER_WHISPER,
    }:
        from buzz.transcriber.whisper_file_transcriber import WhisperFileTranscriber

        return WhisperFileTranscriber(task=task)

    raise ValueError(f"Unknown model type: {model_type}")
