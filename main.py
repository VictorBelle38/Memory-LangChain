# IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory                   # Permite criar Históricos de mensagens
from langchain_core.chat_history import BaseChatMessageHistory                              # Classe base para histórico de mensagens
from langchain_core.runnables.history import RunnableWithMessageHistory                     # Permite gerenciar o histórico de mensagens
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder                  # Permite criar prompts / mensagens
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages   # Mensagens humanas, do sistema e do AI
from langchain_core.runnables import RunnablePassthrough                                    # Permite criar fluxos de execução e reutilizaveis
from operator import itemgetter                                                             # Facilita a extração de valores de dicionários
import transformers


# Carregar as variáveis de ambiente do arquvo .env (para proteger as credenciais)
load_dotenv(find_dotenv())

# Obter a chave da API do GROQ armazenada no arquivo .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicializar o modelo de AI utilizando a API da GROQ
model = ChatGroq(
    model = "gemma2-9b-it",
    groq_api_key = GROQ_API_KEY
)

# Dicionário para armazenar o histórico de mensagens
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    Recura ou cria um histórico de mansagens para uma determinada sesão.
    Isso permite manter o contexto contínuo para diferentes usuários e interações.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Criar um gerenciador de histórico que conecta o modelo ao armazenamento de mensagens
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# Configuração da sessão (Identificador único para cada chat/usuário)
config = {"configurable":{"session_id":"chat1"}}

# Exemplo de interação inicial do usuário
response = with_message_history.invoke(
    [HumanMessage(content="Oi, meu nome é Victor e sou Cientista de Dados.")],
    config=config
)

#Criação de um prompt template para estruturar a entrada do modelo

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Você é um assistente útil. Responda todas as perguntas realizada com precisão ."),
        MessagesPlaceholder(variable_name="messages") #Permitir adicionar mensagens de forma dinamica
    ]

)

#Conectar o modelo ao template de prompt
chain = prompt | model # Usando o LCEL para conectar o prompt ao Modelo

response = chain.invoke(
    {"messages":[HumanMessage(content="Oi, o meu nome é Victor!")]}

)

# Gerenciamento da memoria do Chatbot
trimmer = trim_messages(
    max_tokens = 45, # Define o Limite máximo de tokens para as mensagens evitando a ultrapassagem do consumo de memoria
    strategy = "last", # Define a estrategia de corte das mensagens mais antigas
    token_counter = model, #Modelo realiza a contagem de tokens
    include_system = True, #Inclui mensagens do sistema de histórico
    allow_partial = False, #Evita que as mensagens sejam cortadas parcialmente
    start_on = "human" # Começa a contagem com a mensagem humana
)

#Exemplo de Histórico de Mensagem
messages = [
    SystemMessage(content= "Você é um assistente. Responda todas as perguntas com precisão"),
    HumanMessage(content="Olá, meu nome é Victor"),
    AIMessage(content="Oi Victor! Como posso te auxiliar hoje?"),
    HumanMessage(content="Eu gosto de chocolate branco"),
]

#Aplicar o Limitador e Memoria ao historico
trimmer.invoke(messages)

chain(
    RunnablePassthrough.assign(message=itemgetter("message") | trimmer) # Aplica a otimização do historico
    | prompt # Passa a entrada pelo template de propt
    | model # Passa a entrada pelo modelo
)

response = chain.invoke(
    {
        "messages":[HumanMessage(content="Qual é o chocolate que eu gosto?"),] 
                             
    }
)

print("resposta final:", response)


