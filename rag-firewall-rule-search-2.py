import os
import argparse  # Added for CLI argument handling
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

class FirewallRuleSearcher:
    """Enhanced firewall rule search system using Azure Cognitive Search and GPT-3.5"""
    
    def __init__(self):
        """Initialize Azure services with environment configuration"""
        load_dotenv()
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        )
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            temperature=0.3
        )
        self.vector_store = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
            index_name="firewall-rules-index",
            embedding_function=self.embedding_model
        )
    
    def parse_xml_to_json(self, xml_file_path: str) -> List[Dict[str, Any]]:
        """Converts XML firewall rules to structured JSON format"""
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            rules = []
            
            for rule_elem in root.findall('.//FirewallRule'):
                rule = {
                    'name': self._get_element_text(rule_elem, 'Name'),
                    'description': self._get_element_text(rule_elem, 'Description'),
                    'ip_family': self._get_element_text(rule_elem, 'IPFamily'),
                    'status': self._get_element_text(rule_elem, 'Status'),
                    'position': self._get_element_text(rule_elem, 'Position'),
                    'policy_type': self._get_element_text(rule_elem, 'PolicyType')
                }
                
                if (np := rule_elem.find('NetworkPolicy')):
                    rule.update({
                        'action': self._get_element_text(np, 'Action'),
                        'log_traffic': self._get_element_text(np, 'LogTraffic'),
                        'source_zones': [z.text for z in np.findall('SourceZones/Zone')],
                        'destination_zones': [z.text for z in np.findall('DestinationZones/Zone')],
                        'destination_networks': [n.text for n in np.findall('DestinationNetworks/Network')],
                        'schedule': self._get_element_text(np, 'Schedule'),
                        'web_filter': self._get_element_text(np, 'WebFilter'),
                        'scan_virus': self._get_element_text(np, 'ScanVirus'),
                        'zero_day_protection': self._get_element_text(np, 'ZeroDayProtection'),
                        'application_control': self._get_element_text(np, 'ApplicationControl'),
                        'intrusion_prevention': self._get_element_text(np, 'IntrusionPrevention')
                    })
                rules.append(rule)
            return rules
            
        except (ET.ParseError, FileNotFoundError) as e:
            print(f"⚠️ XML processing error: {e}")
            return []
    
    def _get_element_text(self, parent, tag_name: str) -> str:
        """Safe XML text extraction with fallback to empty string"""
        elem = parent.find(tag_name)
        return elem.text.strip() if elem is not None and elem.text else ""
    
    def create_searchable_documents(self, rules: List[Dict[str, Any]]) -> List[Document]:
        """Transforms JSON rules into vector-search optimized documents"""
        documents = []
        for i, rule in enumerate(rules):
            content = f"""
            {rule.get('name')} | Status: {rule.get('status')}
            Action: {rule.get('action')} | IP: {rule.get('ip_family')}
            Source: {", ".join(rule.get('source_zones', []))}
            Destination: {", ".join(rule.get('destination_zones', []))}
            Networks: {", ".join(rule.get('destination_networks', []))}
            Security: 
            • AV: {rule.get('scan_virus')} 
            • IPS: {rule.get('intrusion_prevention')}
            • Zero-Day: {rule.get('zero_day_protection')}
            """.strip()
            
            metadata = {
                "rule_id": i,
                "rule_name": rule.get('name'),
                "action": rule.get('action'),
                "security_features": {
                    "av": rule.get('scan_virus'),
                    "ips": rule.get('intrusion_prevention'),
                    "zero_day": rule.get('zero_day_protection')
                }
            }
            documents.append(Document(page_content=content, metadata=metadata))
        return documents
    
    def load_and_index_rules(self, xml_file_path: str):
        """End-to-end XML to vector index conversion"""
        if not (rules := self.parse_xml_to_json(xml_file_path)):
            return
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=64,
            separators=["\n", "•", "|"]
        )
        documents = self.create_searchable_documents(rules)
        split_docs = splitter.split_documents(documents)
        
        try:
            self.vector_store.add_documents(split_docs)
            print(f"✅ Successfully indexed {len(split_docs)} rule segments")
        except Exception as e:
            print(f"⛔ Indexing error: {e}")
    
    def search_rules(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Semantic search with GPT-3.5 analysis"""
        try:
            # Vector store returns Document objects containing:
            #   page_content: Human-readable rule summary
            #   metadata: Technical details (action, security features)
            docs = self.vector_store.similarity_search(query, k=top_k)
            
            # Context assembly for GPT-3.5
            context = "\n---\n".join([
                f"RULE {i+1}:\n{doc.page_content}\nMETADATA: {json.dumps(doc.metadata)}" 
                for i, doc in enumerate(docs)
            ])
            
            # GPT-3.5 analysis
            response = self.llm.invoke([
                SystemMessage(content="You're a firewall security analyst. Analyze rules with focus on:"),
                HumanMessage(content=f"Context:\n{context}\n\nQuery: {query}")
            ])
            
            return {
                "answer": response.content,
                "context": context,
                "documents": docs
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def interactive_search(self):
        """User-friendly search interface"""
        print("\n🔥 Firewall Rule Analyzer Ready (type 'exit' to quit)")
        while (query := input("\n🔍 Query: ").strip().lower()) not in {"exit", "quit", ""}:
            if results := self.search_rules(query):
                print(f"\n🛡️ AI ANALYSIS:\n{'-'*40}\n{results['answer']}\n")
    
    def __del__(self):
        """Resource cleanup"""
        gc.collect()

# Added entry point for script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Firewall Rule Semantic Search")
    parser.add_argument("xml_file", help="Path to firewall rules XML file")
    
    args = parser.parse_args()

    searcher = FirewallRuleSearcher()
    searcher.load_and_index_rules(args.xml_file)
    searcher.interactive_search()
