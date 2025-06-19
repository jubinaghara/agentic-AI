# Simple LangChain Example: Recipe Generator
# This example shows how LangChain makes AI applications easier to build
# Updated to use MODERN LangChain syntax (no deprecation warnings!)

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate

class SimpleRecipeGenerator:
    """
    A simple example that shows LangChain's core concepts:
    1. Prompt Templates (reusable prompts with variables)
    2. LLM Chains (connecting prompts to AI models)
    3. How this makes AI applications easier to manage
    """
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Setup Azure OpenAI - same as before
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            temperature=0.7,  # Some creativity for recipes
            max_tokens=400
        )
        
        # 📝 STEP 1: Create a Prompt Template
        # This is like a fill-in-the-blank form for our AI prompts
        # Instead of writing the same prompt structure over and over,
        # we create a template with variables that can be filled in
        self.recipe_template = """
        You are a professional chef with 20 years of experience.
        
        Create a simple recipe using these ingredients: {ingredients}
        Cuisine style: {cuisine}
        Cooking time available: {time_limit}
        
        Please provide:
        1. Recipe name
        2. Ingredients list (with quantities)
        3. Step-by-step instructions
        4. Cooking time
        
        Keep it simple and easy to follow!
        """
        
        # 🔧 STEP 2: Create the PromptTemplate object
        # This tells LangChain what variables our template needs
        self.prompt = PromptTemplate(
            input_variables=["ingredients", "cuisine", "time_limit"],  # These are our variables
            template=self.recipe_template  # This is our template text
        )
        
        # ⛓️ STEP 3: Create a Chain using Modern LangChain Syntax
        # NEW WAY: Use the pipe operator (|) to chain components
        # This is the modern, recommended approach in LangChain
        # Think of it as: Template | AI Model = Chain
        self.recipe_chain = self.prompt | self.llm
        
        print("✅ Recipe Generator initialized!")
        print("🔗 LangChain components ready:")
        print("   📝 Prompt Template - for consistent prompt structure")
        print("   ⛓️  Modern Chain - uses pipe operator (|) syntax")
    
    def generate_recipe(self, ingredients, cuisine="any", time_limit="30 minutes"):
        """
        Generate a recipe using our LangChain setup
        
        This method shows how easy it is to use LangChain:
        1. Just call the chain with our variables
        2. LangChain handles the rest (formatting, AI call, response)
        """
        
        print(f"\n🍳 Generating recipe...")
        print(f"   Ingredients: {ingredients}")
        print(f"   Cuisine: {cuisine}")
        print(f"   Time limit: {time_limit}")
        
        try:
            # 🚀 This is where the magic happens!
            # Modern LangChain syntax using .invoke()
            # LangChain will:
            # 1. Take our template
            # 2. Fill in the variables
            # 3. Send it to Azure OpenAI
            # 4. Return the response
            response = self.recipe_chain.invoke({
                "ingredients": ingredients,
                "cuisine": cuisine,
                "time_limit": time_limit
            })
            
            # Extract the content from the response
            recipe_text = response.content
            
            print("\n" + "="*60)
            print("🍽️  YOUR RECIPE:")
            print("="*60)
            print(recipe_text)
            print("="*60)
            
            return recipe_text
            
        except Exception as e:
            print(f"❌ Error generating recipe: {e}")
            return None
    
def main():
    """
    Main function to demonstrate the Recipe Generator
    """
    
    print("🚀 Simple LangChain Example: Recipe Generator")
    print("=" * 60)
    print("This example shows:")
    print("📝 Prompt Templates - reusable prompts with variables")
    print("⛓️  Modern Chains - using pipe (|) operator")
    print("🔄 How LangChain simplifies AI application development")
    print()
    
    # Create our recipe generator
    generator = SimpleRecipeGenerator()
    
    # Example 1: Italian pasta dish
    print("\n🍝 EXAMPLE 1: Italian Pasta")
    generator.generate_recipe(
        ingredients="tomatoes, garlic, basil, pasta, olive oil",
        cuisine="Italian",
        time_limit="20 minutes"
    )
    
    # Example 2: Quick Asian stir-fry
    print("\n🥢 EXAMPLE 2: Asian Stir-fry")
    generator.generate_recipe(
        ingredients="chicken, broccoli, soy sauce, ginger, rice",
        cuisine="Asian",
        time_limit="15 minutes"
    )
    
    print("\n" + "="*60)
    print("🎯 KEY TAKEAWAY:")
    print("="*60)
    print("""
    LangChain makes AI applications easier by providing:
    
    🧩 REUSABLE COMPONENTS: 
       - Templates you can use over and over
       - Chains that connect different AI operations
    
    📝 CLEANER CODE:
       - Less boilerplate code
       - Standardized patterns
       - Easy to modify and extend
    
    🔗 EASY CHAINING:
       - Connect multiple AI operations
       - Output of one becomes input of another
       - Build complex workflows easily
    
    Think of LangChain as "LEGO blocks for AI" - you can build
    complex AI applications by connecting simple, reusable pieces!
    """)

# Run the example
if __name__ == "__main__":
    main()