import speech_recognition as sr
from secretary import TeacherLlama
from vectorManager import VectorManager
from langchain.docstore.document import Document


r = sr.Recognizer()
def humanList():
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
        try:       
            question = r.recognize_google(audio)
            print("You said " + question + "!")
        except:
            print("Sorry could not recognize what you said")
            raise RuntimeError("Something went wrong "+question)

        

result = []
input = "My day is amazing!"
doc = Document(page_content=input,
        metatdata={
            "source": "userinput"
        }
    )
result.append([doc])
vm = VectorManager()
teacher = TeacherLlama()
vm.save_faiss(result)
print (teacher.answer_question("Hi how are you?", vm.load_faiss()))