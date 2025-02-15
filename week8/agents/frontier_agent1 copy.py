# imports

import os
import re
import math
import json
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from testing import Tester
from agents.agent import Agent


class FrontierAgent1(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE

    #MODEL = "gpt-4o-mini"
    def __init__():
        """
        Set up this instance by connecting to OpenAI or DeepSeek, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.log("Initializing Frontier Agent t1s")
        import os
        import io
        import sys
        from dotenv import load_dotenv
        #from openai import OpenAI
        import google.generativeai
        #import anthropic
        from IPython.display import Markdown, display, update_display
        import subprocess
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        import os
        import google.generativeai as genai

        genai.configure(api_key= api_key)
        # Create the model
        generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 16,
        "max_output_tokens": 500,
        "response_mime_type": "text/plain",
        }
        flash = genai.GenerativeModel(model_name="gemini-1.5-flash",
        generation_config=generation_config,)
        self.chat_session = flash.start_chat(history=[  ])

        self.collection = collection

        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.log("Frontier Agent is ready")

    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def generate_message_for_item(self, item, ):
        # Find similar items and their prices
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        vector = self.model.encode([item])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        #return documents, prices

        # below is my approach, above this code
        # results = self.collection.query(query_embeddings=vector(item).astype(float).tolist(), n_results=5)
        # documents = results['documents'][0][:]
        # prices = [m['price'] for m in results['metadatas'][0][:]]

        # Create the context message for similar items
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(documents, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        
        # Construct the user prompt with the context and item-specific question
        user_prompt = "You estimate prices of items. Reply only with the price, no explanation\n"
        user_prompt += "Do NOT provide esitmates, ranages, or suggestions. Just provide your estimated price\n"
        user_prompt += "Try to be as accurate as possible, based on the examples and information you have"
        user_prompt += message
        user_prompt += "And now the question for you:\n\n"
        user_prompt += item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
        return user_prompt
    
    def price(self, item: str) -> float:
        """
        Make a call to OpenAI or DeepSeek to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price
        """
        a = self.generate_message_for_item(item)
        response = self.chat_session.send_message(a)
        # self.price = self.get_price(response.text)  # Assign calculated price to the instance attribute
        # return self.price
        response1 = self.get_price(response.text)
        return response1


   
