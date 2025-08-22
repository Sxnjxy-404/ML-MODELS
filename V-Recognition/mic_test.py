import speech_recognition as sr

def mic_test():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Speak something (Mic Test)...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            print("🔍 Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"✅ You said: {text}")
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
        except sr.RequestError as e:
            print(f"⚠️ Could not request results; {e}")

if __name__ == "__main__":
    mic_test()
