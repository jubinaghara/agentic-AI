import os
import gc
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import AzureSearch
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue

class FirewallRuleSearcher:
    """
    A class to search firewall rules using Azure Vector Store and GPT-3.5.
    Converts XML firewall rules to JSON format and enables semantic search.
    """
    
    def __init__(self):
        """Initialize the FirewallRuleSearcher with Azure services."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize embedding model for vector representations
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        
        # Initialize chat model (GPT-3.5-turbo)
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=0.3
        )
        
        # Initialize Azure Search vector store
        self.vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
            index_name="firewall-rules-index",
            embedding_function=self.embedding_model
        )
    
    def parse_xml_to_json(self, xml_file_path: str) -> List[Dict[str, Any]]:
        """
        Parse XML firewall rules and convert them to JSON format.
        
        Args:
            xml_file_path (str): Path to the XML file containing firewall rules
            
        Returns:
            List[Dict]: List of firewall rules in JSON format
        """
        print("📄 Parsing XML firewall rules...")
        
        try:
            # Parse the XML file
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            rules = []
            
            # Extract each FirewallRule element
            for rule_elem in root.findall('.//FirewallRule'):
                rule = {}
                
                # Extract basic rule information
                rule['name'] = self._get_element_text(rule_elem, 'Name')
                rule['description'] = self._get_element_text(rule_elem, 'Description')
                rule['ip_family'] = self._get_element_text(rule_elem, 'IPFamily')
                rule['status'] = self._get_element_text(rule_elem, 'Status')
                rule['position'] = self._get_element_text(rule_elem, 'Position')
                rule['policy_type'] = self._get_element_text(rule_elem, 'PolicyType')
                
                # Extract NetworkPolicy details
                network_policy = rule_elem.find('NetworkPolicy')
                if network_policy is not None:
                    rule['action'] = self._get_element_text(network_policy, 'Action')
                    rule['log_traffic'] = self._get_element_text(network_policy, 'LogTraffic')
                    
                    # Extract source zones
                    source_zones = network_policy.find('SourceZones')
                    if source_zones is not None:
                        rule['source_zones'] = [zone.text for zone in source_zones.findall('Zone')]
                    
                    # Extract destination zones
                    dest_zones = network_policy.find('DestinationZones')
                    if dest_zones is not None:
                        rule['destination_zones'] = [zone.text for zone in dest_zones.findall('Zone')]
                    
                    # Extract destination networks
                    dest_networks = network_policy.find('DestinationNetworks')
                    if dest_networks is not None:
                        rule['destination_networks'] = [net.text for net in dest_networks.findall('Network')]
                    
                    # Extract security settings
                    rule['schedule'] = self._get_element_text(network_policy, 'Schedule')
                    rule['web_filter'] = self._get_element_text(network_policy, 'WebFilter')
                    rule['scan_virus'] = self._get_element_text(network_policy, 'ScanVirus')
                    rule['zero_day_protection'] = self._get_element_text(network_policy, 'ZeroDayProtection')
                    rule['application_control'] = self._get_element_text(network_policy, 'ApplicationControl')
                    rule['intrusion_prevention'] = self._get_element_text(network_policy, 'IntrusionPrevention')
                
                rules.append(rule)
            
            print(f"✅ Parsed {len(rules)} firewall rules successfully")
            return rules
            
        except ET.ParseError as e:
            print(f"❌ Error parsing XML: {e}")
            return []
        except FileNotFoundError:
            print(f"❌ File not found: {xml_file_path}")
            return []
    
    def _get_element_text(self, parent, tag_name):
        """Helper method to safely get text from XML element."""
        element = parent.find(tag_name)
        return element.text if element is not None and element.text else ""
    
    def create_searchable_documents(self, rules: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert firewall rules to searchable documents with rich text descriptions.
        
        Args:
            rules (List[Dict]): List of firewall rules in JSON format
            
        Returns:
            List[Document]: List of LangChain documents for vector store
        """
        print("🔄 Creating searchable documents...")
        
        documents = []
        
        for i, rule in enumerate(rules):
            # Create a comprehensive text description of the rule
            description = f"""
                Firewall Rule: {rule.get('name', 'Unnamed')}
                Description: {rule.get('description', 'No description')}
                Status: {rule.get('status', 'Unknown')}
                Action: {rule.get('action', 'Unknown')}
                IP Family: {rule.get('ip_family', 'Unknown')}

                Source Zones: {', '.join(rule.get('source_zones', []))}
                Destination Zones: {', '.join(rule.get('destination_zones', []))}
                Destination Networks: {', '.join(rule.get('destination_networks', []))}

                Security Settings:
                - Schedule: {rule.get('schedule', 'Not specified')}
                - Web Filter: {rule.get('web_filter', 'Not specified')}
                - Virus Scanning: {rule.get('scan_virus', 'Not specified')}
                - Zero Day Protection: {rule.get('zero_day_protection', 'Not specified')}
                - Application Control: {rule.get('application_control', 'Not specified')}
                - Intrusion Prevention: {rule.get('intrusion_prevention', 'Not specified')}
                - Log Traffic: {rule.get('log_traffic', 'Not specified')}

                Position: {rule.get('position', 'Not specified')}
                Policy Type: {rule.get('policy_type', 'Not specified')}
                            """.strip()
            
            # Create metadata with the full rule data
            metadata = {
                "rule_id": i,
                "rule_name": rule.get('name', 'Unnamed'),
                "source_zones": rule.get('source_zones', []),
                "destination_zones": rule.get('destination_zones', []),
                "action": rule.get('action', 'Unknown'),
                "status": rule.get('status', 'Unknown'),
                "full_rule": json.dumps(rule, indent=2)
            }
            
            # Create document
            doc = Document(page_content=description, metadata=metadata)
            documents.append(doc)
        
        print(f"✅ Created {len(documents)} searchable documents")
        return documents
    
    def load_and_index_rules(self, xml_file_path: str):
        """
        Load firewall rules from XML file and index them in the vector store.
        
        Args:
            xml_file_path (str): Path to the XML file containing firewall rules
        """
        print("🚀 Starting firewall rules indexing process...")
        
        # Step 1: Parse XML to JSON
        rules = self.parse_xml_to_json(xml_file_path)
        if not rules:
            print("❌ No rules found or parsing failed")
            return
        
        # Step 2: Create searchable documents
        documents = self.create_searchable_documents(rules)
        
        # Step 3: Split documents if they exceed size limits
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Keep chunks smaller for better matching
            chunk_overlap=100,
            separators=["\n\n", "\n", ":", "-", " "]
        )
        
        # Split large documents
        split_documents = []
        for doc in documents:
            if len(doc.page_content) > 1000:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    split_doc = Document(page_content=chunk, metadata=doc.metadata)
                    split_documents.append(split_doc)
            else:
                split_documents.append(doc)
        
        print(f"📦 Created {len(split_documents)} document chunks for indexing")
        
        # Step 4: Add documents to vector store
        print("🔍 Adding documents to Azure Vector Store...")
        try:
            self.vector_store.add_documents(split_documents)
            print("✅ Successfully indexed all firewall rules")
        except Exception as e:
            print(f"❌ Error indexing documents: {e}")
    
    def search_rules(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search for firewall rules based on user query.
        
        Args:
            query (str): User search query
            top_k (int): Number of top results to return
            
        Returns:
            Dict: Search results with AI-generated answer and retrieved documents
        """
        print(f"🔍 Searching for: '{query}'")
        
        try:
            # Retrieve relevant documents from vector store
            relevant_docs = self.vector_store.similarity_search(query, k=top_k)
            
            if not relevant_docs:
                return {
                    "answer": "No matching firewall rules found for your query.",
                    "retrieved_docs": [],
                    "context": ""
                }
            
            # Prepare context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Create messages for GPT-3.5
            messages = [
                SystemMessage(content="""You are a firewall security expert assistant. 
                Help users understand firewall rules by analyzing the provided context. 
                Focus on matching rules, explaining their purpose, and highlighting security implications.
                Be specific about source zones, destination zones, actions, and security settings."""),
                
                HumanMessage(content=f"""Based on the following firewall rules context, answer the user's question.
                Provide a clear, detailed explanation of matching rules and their security implications.

Context:
{context}

User Question: {query}

Please provide:
1. Summary of closest matching rules
""")
            ]
            
            # Generate AI response
            prompt = ChatPromptValue(messages=messages)
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content,
                "retrieved_docs": relevant_docs,
                "context": context,
                "num_results": len(relevant_docs)
            }
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return {
                "answer": f"Error occurred during search: {str(e)}",
                "retrieved_docs": [],
                "context": ""
            }
    
    def print_search_results(self, results: Dict[str, Any]):
        """Print formatted search results."""
        print("\n" + "="*80)
        print("🎯 AI ANALYSIS:")
        print("="*80)
        print(results["answer"])
        
       # print(f"\n📄 RETRIEVED CONTEXT ({results.get('num_results', 0)} matches):")
       # print("-"*80)
        
        # for i, doc in enumerate(results["retrieved_docs"], 1):
        #     print(f"\n--- Match {i} ---")
        #     print(f"Rule Name: {doc.metadata.get('rule_name', 'Unknown')}")
        #     print(f"Source Zones: {doc.metadata.get('source_zones', [])}")
        #     print(f"Destination Zones: {doc.metadata.get('destination_zones', [])}")
        #     print(f"Action: {doc.metadata.get('action', 'Unknown')}")
        #     print(f"Status: {doc.metadata.get('status', 'Unknown')}")
        #     print(f"\nContent Preview:")
        #     print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
        
        # print("\n" + "="*80 + "\n")
    
    def interactive_search(self):
        """Start interactive search session."""
        print("🔥 Firewall Rule Search Assistant")
        print("="*50)
        print("Enter queries like:")
        print("- 'show rules matching source zone as LAN'")
        print("- 'find rules with virus scanning enabled'")
        print("- 'rules allowing traffic to WAN'")
        print("- 'rules with intrusion prevention'")
        print("\nType 'exit' or 'quit' to stop\n")
        
        while True:
            query = input("🔍 Search Query: ").strip()
            
            if query.lower() in {"exit", "quit", ""}:
                print("👋 Goodbye!")
                break
            
            results = self.search_rules(query)
            self.print_search_results(results)
    
    def cleanup(self):
        """Clean up resources."""
        try:
            del self.vector_store
            gc.collect()
            print("🧹 Resources cleaned up")
        except:
            pass

# Demonstration function
def demonstrate_vector_store_output():
    """
    Demonstrate what gets fed to GPT-3.5 by showing vector store output.
    """
    print("\n" + "="*80)
    print("📊 VECTOR STORE OUTPUT DEMONSTRATION")
    print("="*80)
    
    # Sample firewall rule in the format that gets fed to GPT-3.5
    sample_context = """
Firewall Rule: Clone_#Default_Network_Policy
Description: 
Status: Enable
Action: Accept
IP Family: IPv4

Source Zones: LAN
Destination Zones: WAN
Destination Networks: Internet IPv4 group

Security Settings:
- Schedule: All The Time
- Web Filter: None
- Virus Scanning: Disable
- Zero Day Protection: Disable
- Application Control: None
- Intrusion Prevention: None
- Log Traffic: Disable

Position: Top
Policy Type: Network

---

Firewall Rule: #Default_Network_Policy
Description: 
Status: Enable
Action: Accept
IP Family: IPv4

Source Zones: LAN
Destination Zones: WAN
Destination Networks: Internet IPv4 group

Security Settings:
- Schedule: All The Time
- Web Filter: Allow All
- Virus Scanning: Enable
- Zero Day Protection: Enable
- Application Control: Allow All
- Intrusion Prevention: generalpolicy
- Log Traffic: Disable

Position: After
Policy Type: Network
    """
    
    print("This is the formatted context that gets sent to GPT-3.5:")
    print("-" * 60)
    print(sample_context)
    print("-" * 60)
    print("\nGPT-3.5 receives this context along with the user query")
    print("and provides intelligent analysis of the firewall rules.")
    print("="*80 + "\n")

# Main execution
def main():
    """Main function to run the firewall rule search system."""
    
    # Demonstrate vector store output first
    demonstrate_vector_store_output()
    
    # Initialize the search system
    searcher = FirewallRuleSearcher()
    
    # Define the XML file path (assuming it's in the same folder)
    xml_file_path = "firewall_rules.xml"
    
    try:
        # Load and index the firewall rules
        searcher.load_and_index_rules(xml_file_path)
        
        # Start interactive search
        searcher.interactive_search()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up resources
        searcher.cleanup()

if __name__ == "__main__":
    main()