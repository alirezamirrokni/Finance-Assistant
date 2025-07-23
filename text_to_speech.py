from gtts import gTTS
from pydub import AudioSegment
import os

class TextToSpeech:
    def __init__(
        self,
        lang: str = "en",
        tld: str = "com",
        lang_check: bool = True,
        speed: float = 1.0,
        volume_gain_db: float = 0.0,
        pitch_semitones: float = 0.0,
        output_format: str = None,
    ):
        """
        :param lang: language code (e.g. "en", "es")
        :param tld: Google TTS top-level domain (e.g. "com", "co.uk")
        :param lang_check: verify that lang is supported
        :param speed: playback speed multiplier (1.0 = normal)
        :param volume_gain_db: gain in dB (positive to boost, negative to reduce)
        :param pitch_semitones: pitch shift, in semitones
        :param output_format: "mp3" or "wav"; if None, inferred from filename
        """
        self.lang = lang
        self.tld = tld
        self.lang_check = lang_check
        self.speed = speed
        self.volume_gain_db = volume_gain_db
        self.pitch_semitones = pitch_semitones
        self.output_format = output_format

    def save(self, text: str, filename: str = "output.mp3") -> str:
        """
        Convert `text` to speech and save to `filename`.
        Returns the path to the saved file.
        """
        # determine format
        fmt = (self.output_format or os.path.splitext(filename)[1].lstrip(".")).lower()
        if fmt not in ("mp3", "wav"):
            raise ValueError("output_format/filename extension must be mp3 or wav")

        tmp_mp3 = "_tmp_tts.mp3"
        # 1️⃣ generate TTS
        tts = gTTS(text=text, lang=self.lang, tld=self.tld, lang_check=self.lang_check)
        tts.save(tmp_mp3)

        # 2️⃣ load and transform
        audio = AudioSegment.from_file(tmp_mp3, format="mp3")

        if self.speed != 1.0:
            new_frame_rate = int(audio.frame_rate * self.speed)
            audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
            audio = audio.set_frame_rate(audio.frame_rate)

        if self.volume_gain_db != 0.0:
            audio = audio.apply_gain(self.volume_gain_db)

        if self.pitch_semitones != 0.0:
            ratio = 2 ** (self.pitch_semitones / 12)
            pitched = audio._spawn(audio.raw_data, overrides={
                "frame_rate": int(audio.frame_rate * ratio)
            })
            audio = pitched.set_frame_rate(audio.frame_rate)

        # 3️⃣ export
        out_file = filename if filename.endswith(f".{fmt}") else f"{filename}.{fmt}"
        audio.export(out_file, format=fmt)

        # cleanup
        os.remove(tmp_mp3)
        print(f"✅ Saved speech to {out_file}")
        return out_file
