from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from log.logger import Logger


class TeacherLlama(object):

    def __init__(self):
        self.template = """Answer yes or no usign the following context
                            {contexto}

                            Question: {question}
                        """
        self.prompt = ChatPromptTemplate.from_template(template=self.template)
        self.llm = ChatOllama(model='llama2')
        self.parser = StrOutputParser()

    def answer_question(self, question: str, retriver):
        chain = {"contexto" : retriver, "question" : RunnablePassthrough()} | self.prompt | self.llm | self.parser    
        print('Realizando pregunta '+question)
        result = chain.invoke(question)
        print('Respuesta obtenida: '+result)
        return result
    