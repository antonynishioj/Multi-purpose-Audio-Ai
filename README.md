# Multi-purpose-Audio-Ai
This code performs the following functions with gradio user interface - predicts emotion using audio, detects language, transcribe audio to text, Multi speaker recognition, audio stem separation(splits main audio into various audio according to instruments and vocal) <br />
<br />
## Libraries used:<br />
1.spleeter - audio stem seperation <br />
2.tensorflow and librosa - emotion detection using audio <br />
3.whisper - transcribe audio to text and detect language <br />
4.whisperx - multi speaker recognition <br />
5.gradio - user interface <br />
<br />
## pretrained model(speechemotion.h5):<br />
I have provided the emotion detection pre trained model .h5 file "speechemotion.h5" which can be used to detect emotions in speech like angry, disgust, fear, happy, neutral, calm, sad <br />
<br />
## problems you may face:<br />
while installing spleeter library it deletes some libraries and installs same libraries older or new version, so incase if you face any error like that try the code by removing spleeter and audio stem separation but if that part is required you can try implementing it alone in another file.
