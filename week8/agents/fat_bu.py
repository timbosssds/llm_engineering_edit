import os
import re
import math
import json
import logging
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from testing import Tester
from agents.agent import Agent
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
genai.configure(api_key= api_key)


# Configure logging
logging.basicConfig(level=logging.INFO, format="\033[94m%(name)s:\033[0m %(message)s")
logger = logging.getLogger("fat")

class fat(Agent):

    name = "fat"
    color = Agent.BLUE
    
    def __init__(self, collection):
        logger.info("Initializing Fat class")
        self.collection = collection
        logger.info("Initializing Fat class with collection")
        
    
    def gem_llm(self, message: str) -> str:
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )
        
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(message)
        logger.info("Received response from LLM")
        return response.text
    
    def description(self, item: Item) -> str:
        text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
        logger.info("Extracted description from item")
        return text.split("\n\nPrice is $")[0]
    
    def get_price(self, s: str) -> float:
        s = s.replace('$', '').replace(',', '')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        price = float(match.group()) if match else 0.0
        logger.info(f"Extracted price: {price}")
        return price
    
    def description(self, item: Item) -> str:
        text = item.prompt.replace("How much does this cost to the nearest dollar?\n\n", "")
        logger.info("Extracted description from item")
        return text.split("\n\nPrice is $")[0]

    def vector(self, item):
        modelembed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return modelembed.encode([self.description(item)])
    
    def generate_message_for_item(self, item: Item) -> str:
        DB = "products_vectorstore"
        client = chromadb.PersistentClient(path=DB)
        collection = client.get_or_create_collection('products')
        results = collection.query(query_embeddings=self.vector(item).astype(float).tolist(), n_results=5)  # ✅ Use self.collection
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(documents, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        
        user_prompt = ("You estimate prices of items. Reply only with the price, no explanation.\n"
                       "Do NOT provide estimates, ranges, or suggestions. Just provide your estimated price.\n"
                       "Try to be as accurate as possible, based on the examples and information you have.\n"
                       + message +
                       "And now the question for you:\n\n" +
                       item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", ""))

        logger.info("Generated message for item")
        return user_prompt
    
    # def gpt_4o_mini_rag(self, item: Item) -> float:
    #     message = self.generate_message_for_item(item)
    #     response = self.gem_llm(message)
    #     price = self.get_price(response)
    #     logger.info(f"Estimated price: {price}")
    #     return price

    def price(self, item: Item) -> float:
        message = self.generate_message_for_item(item)
        response = self.gem_llm(message)
        price = self.get_price(response)
        logger.info(f"Estimated price: {price}")
        return price

      
