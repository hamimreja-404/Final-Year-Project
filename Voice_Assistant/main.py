import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser

class VoiceAssistant:
    """
    A class to represent a voice asqqq  sistant that responds to voice commands.
    """

    def __init__(self, wake_word="hello computer"):
        """
        Initializes the VoiceAssistant with a recognizer, a text-to-speech engine,
        and sets the wake word.
        """
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.wake_word = wake_word.lower()

        # Configure the TTS voice
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[1].id) # Index 1 is often a female voice
        self.tts_engine.setProperty('rate', 180) # Speed of speech

    def speak(self, text):
        """
        Converts a string of text into speech.
        """
        print(f"🤖 Assistant: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def listen(self, timeout=5):
        """
        Listens for audio input from the microphone and tries to recognize it.
        Returns the recognized text in lowercase.
        """
        with sr.Microphone() as source:
            print("Listening...")
            # Adjust for ambient noise to improve recognition accuracy
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                # Listen for audio with a timeout
                audio = self.recognizer.listen(source, timeout=timeout)
                print("Recognizing...")
                # Use Google's speech recognition
                command = self.recognizer.recognize_google(audio)
                print(f"👤 User: {command}")
                return command.lower()
            except sr.WaitTimeoutError:
                print("Listening timed out while waiting for phrase to start.")
                return ""
            except sr.UnknownValueError:
                # This error is raised when speech is unintelligible
                return ""
            except sr.RequestError:
                # This error is for issues with the speech recognition service
                self.speak("Sorry, I'm having trouble connecting to my speech service.")
                return ""
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return ""

    def process_command(self, command):
        """
        Processes the given command and performs the appropriate action.
        """
        if not command:
            return

        # --- Add your custom commands here ---
        if "what is your name" in command:
            self.speak("I am a voice assistant, created to help you.")

        elif "what time is it" in command:
            now = datetime.datetime.now().strftime("%I:%M %p")
            self.speak(f"The current time is {now}.")

        elif "open google" in command:
            self.speak("Opening Google.")
            webbrowser.open("https://www.google.com")

        elif "search for" in command:
            # Extracts the search query from the command
            query = command.replace("search for", "").strip()
            if query:
                self.speak(f"Here are the results for {query}.")
                webbrowser.open(f"https://www.google.com/search?q={query}")
            else:
                self.speak("What would you like me to search for?")

        else:
            self.speak("Sorry, I don't know how to do that yet.")


    def run(self):
        """
        The main loop for the voice assistant.
        It continuously listens for the wake word and then for commands.
        """
        self.speak("Hello! I'm ready to help. Just say my wake word.")
        print(f"Wake word is: '{self.wake_word}'")
        
        while True:
            # First, listen for the wake word
            wake_command = self.listen()
            if self.wake_word in wake_command:
                self.speak("I'm listening. How can I help?")
                # Once awake, listen for the actual command
                command = self.listen(timeout=8) # Longer timeout for the command
                if "goodbye" in command or "exit" in command or "quit" in command:
                    self.speak("Goodbye!")
                    break
                self.process_command(command)
                self.speak("Say the wake word again if you need anything else.")


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
