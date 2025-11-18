#!flask/bin/python
import argparse
import io
import json
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Union
from urllib.parse import parse_qs

from flask import Flask, render_template, render_template_string, request, send_file, jsonify

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.api import TTS


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    return parser


# parse the args
args = create_argparser().parse_args()

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

if args.list_models:
    manager.list_models()
    sys.exit()

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit()

# CASE2: load pre-trained model paths
if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

# CASE3: set custom model paths
if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

# load models
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=speakers_file_path,
    tts_languages_file=None,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=args.use_cuda,
)

# Initialize TTS API for complete functionality
tts = TTS(model_name=args.model_name, gpu=args.use_cuda) if args.model_name else TTS(
    model_path=model_path, config_path=config_path, vocoder_path=vocoder_path, vocoder_config_path=vocoder_config_path, gpu=args.use_cuda
)

vc_tts = None  # For TTS with voice conversion

use_multi_speaker = hasattr(synthesizer.tts_model, "num_speakers") and (
    synthesizer.tts_model.num_speakers > 1 or synthesizer.tts_speakers_file is not None
)
speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)

use_multi_language = hasattr(synthesizer.tts_model, "num_languages") and (
    synthesizer.tts_model.num_languages > 1 or synthesizer.tts_languages_file is not None
)
language_manager = getattr(synthesizer.tts_model, "language_manager", None)

# TODO: set this from SpeakerManager
use_gst = synthesizer.tts_config.get("use_gst", False)
app = Flask(__name__)


def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        show_details=args.show_details,
        use_multi_speaker=use_multi_speaker,
        use_multi_language=use_multi_language,
        speaker_ids=speaker_manager.name_to_id if speaker_manager is not None else None,
        language_ids=language_manager.name_to_id if language_manager is not None else None,
        use_gst=use_gst,
    )


@app.route("/details")
def details():
    if args.config_path is not None and os.path.isfile(args.config_path):
        model_config = load_config(args.config_path)
    else:
        if args.model_name is not None:
            model_config = load_config(config_path)

    if args.vocoder_config_path is not None and os.path.isfile(args.vocoder_config_path):
        vocoder_config = load_config(args.vocoder_config_path)
    else:
        if args.vocoder_name is not None:
            vocoder_config = load_config(vocoder_config_path)
        else:
            vocoder_config = None

    return render_template(
        "details.html",
        show_details=args.show_details,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args.__dict__,
    )


lock = Lock()


@app.route("/api/tts", methods=["GET", "POST"])
def tts():
    with lock:
        text = request.headers.get("text") or request.values.get("text", "")
        speaker_idx = request.headers.get("speaker-id") or request.values.get("speaker_id", "")
        language_idx = request.headers.get("language-id") or request.values.get("language_id", "")
        style_wav = request.headers.get("style-wav") or request.values.get("style_wav", "")
        style_wav = style_wav_uri_to_dict(style_wav)

        print(f" > Model input: {text}")
        print(f" > Speaker Idx: {speaker_idx}")
        print(f" > Language Idx: {language_idx}")
        wavs = synthesizer.tts(text, speaker_name=speaker_idx, language_name=language_idx, style_wav=style_wav)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


# Basic MaryTTS compatibility layer


@app.route("/locales", methods=["GET"])
def mary_tts_api_locales():
    """MaryTTS-compatible /locales endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string("{{ locale }}\n", locale=model_details[1])


@app.route("/voices", methods=["GET"])
def mary_tts_api_voices():
    """MaryTTS-compatible /voices endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string(
        "{{ name }} {{ locale }} {{ gender }}\n", name=model_details[3], locale=model_details[1], gender="u"
    )


@app.route("/process", methods=["GET", "POST"])
def mary_tts_api_process():
    """MaryTTS-compatible /process endpoint"""
    with lock:
        if request.method == "POST":
            data = parse_qs(request.get_data(as_text=True))
            # NOTE: we ignore param. LOCALE and VOICE for now since we have only one active model
            text = data.get("INPUT_TEXT", [""])[0]
        else:
            text = request.args.get("INPUT_TEXT", "")
        print(f" > Model input: {text}")
        wavs = synthesizer.tts(text)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


# Comprehensive API endpoints for full model interaction

@app.route("/api/models", methods=["GET"])
def list_models():
    """List all available TTS models"""
    try:
        models = manager.list_tts_models()
        return jsonify({"models": models, "vc_models": manager.list_vc_models()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/info", methods=["GET"])
def current_model_info():
    """Get information about the current loaded model"""
    try:
        info = {
            "model_name": getattr(tts, 'model_name', args.model_name),
            "is_multi_speaker": tts.is_multi_speaker,
            "is_multi_lingual": tts.is_multi_lingual,
            "speakers": tts.speakers,
            "languages": tts.languages
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model/switch", methods=["POST"])
def switch_model():
    """Switch to a different model (simplified version, restarts included)"""
    try:
        global tts, synthesizer
        new_model = request.json.get("model_name")
        if not new_model:
            return jsonify({"error": "model_name required"}), 400

        # Load new TTS model
        tts = TTS(model_name=new_model, gpu=args.use_cuda)
        return jsonify({"message": f"Switched to {new_model}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/vc", methods=["POST"])
def voice_conversion():
    """Voice conversion between audio files"""
    try:
        source_file = request.files.get("source_wav")
        target_file = request.files.get("target_wav")

        if not source_file or not target_file:
            return jsonify({"error": "Both source_wav and target_wav files are required"}), 400

        # Save temp files
        source_path = f"/tmp/source_{hash(source_file.filename)}.wav"
        target_path = f"/tmp/target_{hash(target_file.filename)}.wav"
        source_file.save(source_path)
        target_file.save(target_path)

        # Load VC model if not loaded
        global vc_tts
        if vc_tts is None or not hasattr(vc_tts, 'voice_converter'):
            vc_tts = TTS("voice_conversion_models/multilingual/vctk/freevc24", gpu=args.use_cuda)

        with lock:
            wav = vc_tts.voice_conversion(source_wav=source_path, target_wav=target_path)
            out = io.BytesIO()
            from TTS.utils.audio.numpy_transforms import save_wav
            save_wav(wav=wav, path=out, sample_rate=vc_tts.voice_converter.vc_config.audio.output_sample_rate)
            out.seek(0)

        # Cleanup temp files
        os.remove(source_path)
        os.remove(target_path)

        return send_file(out, mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts_vc", methods=["POST"])
def tts_with_vc():
    """TTS with voice conversion (cloning)"""
    try:
        text = request.form.get("text", "")
        language = request.form.get("language")
        speaker_wav = request.files.get("speaker_wav")

        if not text:
            return jsonify({"error": "text is required"}), 400
        if not speaker_wav:
            return jsonify({"error": "speaker_wav file is required"}), 400

        # Save temp file
        speaker_path = f"/tmp/speaker_{hash(speaker_wav.filename)}.wav"
        speaker_wav.save(speaker_path)

        with lock:
            wav = tts.tts_with_vc(text=text, language=language, speaker_wav=speaker_path)
            out = io.BytesIO()
            from TTS.utils.audio.numpy_transforms import save_wav
            save_wav(wav=wav, path=out, sample_rate=22050)  # Default sample rate
            out.seek(0)

        # Cleanup temp file
        os.remove(speaker_path)

        return send_file(out, mimetype="audio/wav")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/docs", methods=["GET"])
def api_docs():
    """API documentation"""
    docs = {
        "endpoints": {
            "/api/tts": {
                "method": "GET/POST",
                "description": "Text to speech synthesis",
                "parameters": {
                    "text": "Text to synthesize",
                    "speaker_id": "Speaker ID for multi-speaker models",
                    "language_id": "Language ID for multi-lingual models",
                    "style_wav": "Path to style wav or GST tokens JSON"
                }
            },
            "/api/models": {
                "method": "GET",
                "description": "List all available models"
            },
            "/api/model/info": {
                "method": "GET",
                "description": "Get current model information"
            },
            "/api/model/switch": {
                "method": "POST",
                "description": "Switch to a different model",
                "parameters": {
                    "model_name": "Name of the model to switch to"
                }
            },
            "/api/vc": {
                "method": "POST",
                "description": "Voice conversion between audio files",
                "files": {
                    "source_wav": "Source audio file",
                    "target_wav": "Target audio file"
                }
            },
            "/api/tts_vc": {
                "method": "POST",
                "description": "TTS with voice conversion (voice cloning)",
                "parameters": {
                    "text": "Text to synthesize",
                    "language": "Language for synthesis"
                },
                "files": {
                    "speaker_wav": "Speaker audio file for cloning"
                }
            },
            "/process": {
                "method": "GET/POST",
                "description": "MaryTTS-compatible synthesis endpoint",
                "parameters": {
                    "INPUT_TEXT": "Text to synthesize"
                }
            },
            "/voices": {
                "method": "GET",
                "description": "MaryTTS-compatible voices endpoint"
            },
            "/locales": {
                "method": "GET",
                "description": "MaryTTS-compatible locales endpoint"
            }
        }
    }
    return jsonify(docs)


def main():
    app.run(debug=args.debug, host="::", port=args.port)


if __name__ == "__main__":
    main()
