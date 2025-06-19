import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import CommaSeparatedListOutputParser, StructuredOutputParser, ResponseSchema

class LangChainDemo:
    def __init__(self):
        """Initialize LangChain with Azure OpenAI"""
        load_dotenv()
        self.setup_llm()
    
    def setup_llm(self):
        """Setup Azure OpenAI LLM"""
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0.7,
            max_tokens=500
        )
        print("✅ Azure OpenAI LLM initialized!")
    
    def demo_basic_prompt_template(self):
        """Demo 1: Basic Prompt Templates"""
        print("\n" + "="*50)
        print("DEMO 1: BASIC PROMPT TEMPLATES")
        print("="*50)
        
        # Create a prompt template
        template = """
        You are a helpful assistant that explains concepts simply.
        
        Topic: {topic}
        Audience: {audience}
        
        Please explain {topic} in a way that {audience} can understand.
        """
        
        prompt = PromptTemplate(
            input_variables=["topic", "audience"],
            template=template
        )
        
        # Create a chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Test the chain
        result = chain.run(topic="Machine Learning", audience="a 10-year-old child")
        print("🤖 AI Response:")
        print(result)
        
        return result
    
    def demo_chat_prompt_template(self):
        """Demo 2: Chat Prompt Templates"""
        print("\n" + "="*50)
        print("DEMO 2: CHAT PROMPT TEMPLATES")
        print("="*50)
        
        # Create a chat prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a {role} with {years} years of experience."),
            ("human", "Give me advice about: {question}")
        ])
        
        # Create chain
        chain = LLMChain(llm=self.llm, prompt=chat_prompt)
        
        # Test the chain
        result = chain.run(
            role="career counselor",
            years="15",
            question="switching from marketing to data science"
        )
        
        print("🎯 Career Advice:")
        print(result)
        
        return result
    
    def demo_sequential_chains(self):
        """Demo 3: Sequential Chains - Chaining multiple operations"""
        print("\n" + "="*50)
        print("DEMO 3: SEQUENTIAL CHAINS")
        print("="*50)
        
        # First chain: Generate a story idea
        story_template = """
        Create a short story idea about: {topic}
        Include: setting, main character, and conflict.
        Keep it to 2-3 sentences.
        """
        story_prompt = PromptTemplate(
            input_variables=["topic"],
            template=story_template
        )
        story_chain = LLMChain(llm=self.llm, prompt=story_prompt, output_key="story_idea")
        
        # Second chain: Write the opening paragraph
        opening_template = """
        Based on this story idea: {story_idea}
        
        Write an engaging opening paragraph for this story.
        """
        opening_prompt = PromptTemplate(
            input_variables=["story_idea"],
            template=opening_template
        )
        opening_chain = LLMChain(llm=self.llm, prompt=opening_prompt, output_key="opening")
        
        # Combine chains
        overall_chain = SimpleSequentialChain(
            chains=[story_chain, opening_chain],
            verbose=True  # This shows the intermediate steps
        )
        
        # Run the sequential chain
        result = overall_chain.run("space exploration")
        
        print("📚 Final Story Opening:")
        print(result)
        
        return result
    
    def demo_output_parsers(self):
        """Demo 4: Output Parsers - Structure the AI response"""
        print("\n" + "="*50)
        print("DEMO 4: OUTPUT PARSERS")
        print("="*50)
        
        # Comma-separated list parser
        print("📝 Comma-Separated List Parser:")
        list_parser = CommaSeparatedListOutputParser()
        
        list_template = """
        List 5 programming languages suitable for beginners.
        {format_instructions}
        """
        
        list_prompt = PromptTemplate(
            template=list_template,
            input_variables=[],
            partial_variables={"format_instructions": list_parser.get_format_instructions()}
        )
        
        chain = LLMChain(llm=self.llm, prompt=list_prompt)
        output = chain.run({})
        parsed_output = list_parser.parse(output)
        
        print("Raw output:", output)
        print("Parsed output:", parsed_output)
        print("Type:", type(parsed_output))
        
        # Structured output parser
        print("\n🏗️ Structured Output Parser:")
        
        response_schemas = [
            ResponseSchema(name="language", description="programming language name"),
            ResponseSchema(name="difficulty", description="difficulty level (1-10)"),
            ResponseSchema(name="use_case", description="main use case or application"),
            ResponseSchema(name="learning_time", description="estimated learning time")
        ]
        
        structured_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        structured_template = """
        Provide information about Python programming language.
        
        {format_instructions}
        """
        
        structured_prompt = PromptTemplate(
            template=structured_template,
            input_variables=[],
            partial_variables={"format_instructions": structured_parser.get_format_instructions()}
        )
        
        chain = LLMChain(llm=self.llm, prompt=structured_prompt)
        output = chain.run({})
        parsed_output = structured_parser.parse(output)
        
        print("Structured output:")
        for key, value in parsed_output.items():
            print(f"  {key}: {value}")
        
        return parsed_output
    
    def demo_memory_conversation(self):
        """Demo 5: Simple conversation without memory (for basic understanding)"""
        print("\n" + "="*50)
        print("DEMO 5: CONVERSATION SIMULATION")
        print("="*50)
        
        conversation_template = """
        Previous conversation context: {context}
        
        Human: {human_input}
        
        As a friendly AI assistant, respond to the human's message while considering the context.
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "human_input"],
            template=conversation_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Simulate a conversation
        context = ""
        conversations = [
            "Hi! What's your name?",
            "Can you help me learn Python?",
            "What should I start with first?"
        ]
        
        for i, human_input in enumerate(conversations):
            print(f"\n👤 Human: {human_input}")
            
            response = chain.run(context=context, human_input=human_input)
            print(f"🤖 AI: {response}")
            
            # Update context for next iteration
            context += f"Human: {human_input}\nAI: {response}\n"
        
        return context
    
    def run_all_demos(self):
        """Run all LangChain demos"""
        print("🚀 Starting LangChain Demos with Azure OpenAI!")
        print("This will show you the core concepts of LangChain:")
        print("- Prompt Templates")
        print("- Chains")
        print("- Sequential Chains") 
        print("- Output Parsers")
        print("- Conversation Handling")
        
        try:
            self.demo_basic_prompt_template()
            self.demo_chat_prompt_template()
            self.demo_sequential_chains()
            self.demo_output_parsers()
            self.demo_memory_conversation()
            
            print("\n" + "="*50)
            print("🎉 ALL DEMOS COMPLETED!")
            print("="*50)
            print("\nKey LangChain Concepts Demonstrated:")
            print("✅ PromptTemplate - Reusable prompt structures")
            print("✅ LLMChain - Combining prompts with LLMs")
            print("✅ Sequential Chains - Chaining multiple operations")
            print("✅ Output Parsers - Structured response formatting")
            print("✅ Conversation Context - Building conversational AI")
            
        except Exception as e:
            print(f"❌ Error running demos: {e}")

# Example usage
if __name__ == "__main__":
    # Initialize and run demos
    demo = LangChainDemo()
    demo.run_all_demos()
    
    print("\n" + "="*50)
    print("💡 WHAT IS LANGCHAIN?")
    print("="*50)
    print("""
    LangChain is a framework that makes it easy to build applications with LLMs by providing:
    
    🔗 CHAINS: Connect multiple AI operations together
    📝 TEMPLATES: Reusable prompt structures with variables
    🔄 PARSERS: Structure and validate AI responses
    💾 MEMORY: Remember conversation context
    🔧 TOOLS: Integrate with external services
    
    Think of it as "LEGO blocks" for AI applications - you can combine
    simple components to build complex AI-powered systems!
    """)