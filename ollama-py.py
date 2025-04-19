#==============================================================
#Test para reconocer imagenes en LlaVa:7b con ollama localmente
#==============================================================

from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='llava:7b', messages=[
  {
    'role': 'user',
    'content': 'very shortly, what is in this image: ',
    'images' : ['apple-img.jpg']
  },
])
print(response['message']['content'])
