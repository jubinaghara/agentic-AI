import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class AzureOpenAIClient:
    def __init__(self):
        """Initialize Azure OpenAI client with environment variables"""
        # Load environment variables from .env file
        load_dotenv()
        
        self.setup_azure_openai()
    
    def setup_azure_openai(self):
        """Setup Azure OpenAI client using environment variables"""
        # Get credentials from environment variables
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # Validate that all required environment variables are set
        if not all([api_key, endpoint, api_version, deployment_name]):
            raise ValueError("Missing required Azure OpenAI environment variables")
        
        # Initialize the Azure OpenAI client
        self.llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment_name,
            temperature=0.3,
            max_tokens=1000
        )
        
        print("Azure OpenAI client initialized successfully!")
    
    def send_message(self, user_message, system_message=None):
        """Send a message to Azure OpenAI and get response"""
        try:
            messages = []
            
            # Add system message if provided
            if system_message:
                messages.append(SystemMessage(content=system_message))
            
            # Add user message
            messages.append(HumanMessage(content=user_message))
            
            # Get response from Azure OpenAI
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            print(f"Error sending message: {e}")
            return None
    
    def chat_completion(self, prompt, system_prompt=None):
        """Simple chat completion method"""
        return self.send_message(prompt, system_prompt)

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the client
        client = AzureOpenAIClient()
        
        # Example 1: Simple chat
        response = client.chat_completion("Hello! Can you tell me about Python?")
        print("Response:", response)
        
        # Example 2: Chat with system message
        system_msg = "You are a helpful assistant that explains things simply."
        user_msg = "Explain what machine learning is in simple terms."
        
        response = client.send_message(user_msg, system_msg)
        print("\nResponse with system message:", response)
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Install required packages: pip install langchain-openai python-dotenv")
        print("2. Create a .env file with your Azure OpenAI credentials")
        print("3. Set the correct environment variables")