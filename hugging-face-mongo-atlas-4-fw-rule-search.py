# Current code uses semantic (vector-based) search for firewall rules, which is excellent for 
# natural language queries but can be imprecise for structured fields like source_zones. 
# When you search for "source zone as LAN," the embedding model may also retrieve rules mentioning
# WAN (or both), because the semantic similarity between "LAN" and "WAN" is high in vector space. 
# This is why you see results with source zone as WAN when searching for LAN.

import os
import json
import xml.etree.ElementTree as ET
import pymongo
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import login


@dataclass
class FirewallRule:
    """Elegant data structure for firewall rules"""
    name: str
    description: str
    ip_family: str
    status: str
    position: str
    policy_type: str
    action: str
    source_zones: List[str]
    destination_zones: List[str]
    destination_networks: List[str]
    schedule: str
    web_filter: str
    virus_scan: bool
    zero_day_protection: bool
    intrusion_prevention: str
    application_control: str
    log_traffic: bool
    searchable_text: str
    rule_embedding: Optional[List[float]] = None


class FirewallSearchEngine:
    """Apple-style elegant firewall rule search engine"""
    
    def __init__(self, mongo_url: str, hf_token: str, db_name: str = "firewall_db"):
        self._initialize_connections(mongo_url, hf_token, db_name)
        self._setup_ml_model()
    
    def _initialize_connections(self, mongo_url: str, hf_token: str, db_name: str):
        """Initialize database and authentication connections"""
        # MongoDB setup with enhanced security
        self.client = pymongo.MongoClient(mongo_url, tls=True, tlsInsecure=True)
        self.db = self.client[db_name]
        self.firewall_collection = self.db.firewall_rules
        
        # Hugging Face authentication
        if not hf_token or not hf_token.strip():
            raise ValueError("Hugging Face token is required")
        login(token=hf_token)
    
    def _setup_ml_model(self):
        """Load and configure the sentence transformer model"""
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def parse_xml_rules(self, xml_file_path: str) -> List[FirewallRule]:
        """Parse XML firewall rules into structured objects"""
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            rules = []
            
            for rule_element in root.findall('FirewallRule'):
                rule = self._extract_rule_data(rule_element)
                if rule:
                    rules.append(rule)
            
            print(f"✓ Successfully parsed {len(rules)} firewall rules")
            return rules
            
        except ET.ParseError as e:
            print(f"✗ XML parsing error: {e}")
            return []
        except FileNotFoundError:
            print(f"✗ File not found: {xml_file_path}")
            return []
    
    def _extract_rule_data(self, rule_element: ET.Element) -> Optional[FirewallRule]:
        """Extract and structure firewall rule data"""
        try:
            # Basic rule information
            name = self._get_text(rule_element, 'Name')
            description = self._get_text(rule_element, 'Description')
            ip_family = self._get_text(rule_element, 'IPFamily')
            status = self._get_text(rule_element, 'Status')
            position = self._get_text(rule_element, 'Position')
            policy_type = self._get_text(rule_element, 'PolicyType')
            
            # Network policy details
            network_policy = rule_element.find('NetworkPolicy')
            if network_policy is None:
                return None
            
            action = self._get_text(network_policy, 'Action')
            schedule = self._get_text(network_policy, 'Schedule')
            web_filter = self._get_text(network_policy, 'WebFilter')
            intrusion_prevention = self._get_text(network_policy, 'IntrusionPrevention')
            application_control = self._get_text(network_policy, 'ApplicationControl')
            
            # Boolean fields
            virus_scan = self._get_text(network_policy, 'ScanVirus') == 'Enable'
            zero_day_protection = self._get_text(network_policy, 'ZeroDayProtection') == 'Enable'
            log_traffic = self._get_text(network_policy, 'LogTraffic') == 'Enable'
            
            # List fields
            source_zones = self._extract_list_items(network_policy, 'SourceZones/Zone')
            destination_zones = self._extract_list_items(network_policy, 'DestinationZones/Zone')
            destination_networks = self._extract_list_items(network_policy, 'DestinationNetworks/Network')
            
            # Create searchable text for natural language processing
            searchable_text = self._create_searchable_text(
                name, description, action, source_zones, destination_zones,
                destination_networks, web_filter, intrusion_prevention, application_control
            )
            
            return FirewallRule(
                name=name,
                description=description,
                ip_family=ip_family,
                status=status,
                position=position,
                policy_type=policy_type,
                action=action,
                source_zones=source_zones,
                destination_zones=destination_zones,
                destination_networks=destination_networks,
                schedule=schedule,
                web_filter=web_filter,
                virus_scan=virus_scan,
                zero_day_protection=zero_day_protection,
                intrusion_prevention=intrusion_prevention,
                application_control=application_control,
                log_traffic=log_traffic,
                searchable_text=searchable_text
            )
            
        except Exception as e:
            print(f"✗ Error extracting rule data: {e}")
            return None
    
    def _get_text(self, element: ET.Element, tag: str) -> str:
        """Safely extract text from XML element"""
        found = element.find(tag)
        return found.text.strip() if found is not None and found.text else ""
    
    def _extract_list_items(self, element: ET.Element, xpath: str) -> List[str]:
        """Extract list of items from XML using XPath"""
        items = element.findall(xpath)
        return [item.text.strip() for item in items if item.text]
    
    def _create_searchable_text(self, name: str, description: str, action: str,
                              source_zones: List[str], destination_zones: List[str],
                              destination_networks: List[str], web_filter: str,
                              intrusion_prevention: str, application_control: str) -> str:
        """Create comprehensive searchable text for natural language processing"""
        text_parts = [
            f"Rule name: {name}",
            f"Description: {description}",
            f"Action: {action}",
            f"Source zones: {', '.join(source_zones)}",
            f"Destination zones: {', '.join(destination_zones)}",
            f"Destination networks: {', '.join(destination_networks)}",
            f"Web filter: {web_filter}",
            f"Intrusion prevention: {intrusion_prevention}",
            f"Application control: {application_control}"
        ]
        return " | ".join(filter(None, text_parts))
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text"""
        if not text.strip():
            return []
        
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            print(f"✗ Error generating embedding: {e}")
            return []
    
    def import_rules_to_database(self, rules: List[FirewallRule]) -> bool:
        """Import firewall rules to MongoDB with embeddings"""
        try:
            # Clear existing rules
            self.firewall_collection.delete_many({})
            
            # Process rules with embeddings
            documents = []
            for rule in rules:
                # Generate embedding for searchable text
                rule.rule_embedding = self.generate_embedding(rule.searchable_text)
                
                # Convert to dictionary for MongoDB
                doc = asdict(rule)
                documents.append(doc)
            
            # Bulk insert
            if documents:
                result = self.firewall_collection.insert_many(documents)
                print(f"✓ Imported {len(result.inserted_ids)} firewall rules")
                return True
            else:
                print("✗ No valid rules to import")
                return False
                
        except Exception as e:
            print(f"✗ Database import error: {e}")
            return False
    
    def search_rules(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search firewall rules using natural language"""
        try:
            # Generate embedding for search query
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                print("✗ Failed to generate query embedding")
                return []
            
            # Perform vector search
            pipeline = [
                {
                    "$vectorSearch": {
                        "queryVector": query_embedding,
                        "path": "rule_embedding",
                        "numCandidates": 100,
                        "limit": limit,
                        "index": "firewall_rules_search"
                    }
                },
                {
                    "$project": {
                        "name": 1,
                        "description": 1,
                        "action": 1,
                        "source_zones": 1,
                        "destination_zones": 1,
                        "destination_networks": 1,
                        "status": 1,
                        "web_filter": 1,
                        "virus_scan": 1,
                        "zero_day_protection": 1,
                        "intrusion_prevention": 1,
                        "application_control": 1,
                        "searchable_text": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.firewall_collection.aggregate(pipeline))
            print(f"✓ Found {len(results)} matching rules")
            return results
            
        except Exception as e:
            print(f"✗ Search error: {e}")
            return []
    
    def display_search_results(self, results: List[Dict[str, Any]], query: str):
        """Display search results in an elegant format"""
        print(f"\n🔍 Search Results for: '{query}'")
        print("=" * 80)
        
        if not results:
            print("No matching firewall rules found.")
            return
        
        for i, rule in enumerate(results, 1):
            score = rule.get('score', 0)
            print(f"\n{i}. {rule['name']} (Score: {score:.3f})")
            print(f"   Description: {rule['description']}")
            print(f"   Action: {rule['action']} | Status: {rule['status']}")
            print(f"   Source: {', '.join(rule['source_zones'])} → Destination: {', '.join(rule['destination_zones'])}")
            
            if rule.get('web_filter') and rule['web_filter'] != 'None':
                print(f"   Web Filter: {rule['web_filter']}")
            
            security_features = []
            if rule.get('virus_scan'):
                security_features.append("Virus Scan")
            if rule.get('zero_day_protection'):
                security_features.append("Zero Day Protection")
            if rule.get('intrusion_prevention') and rule['intrusion_prevention'] != 'None':
                security_features.append(f"IPS: {rule['intrusion_prevention']}")
            
            if security_features:
                print(f"   Security: {', '.join(security_features)}")
            
            print("-" * 80)


def main():
    """Main execution function"""
    # Load environment variables
    load_dotenv()
    
    # Initialize the search engine
    try:
        engine = FirewallSearchEngine(
            mongo_url=os.getenv("MONGO_DB_URL"),
            hf_token=os.getenv("HUGGING_FACE_ACCESS_TOKEN")
        )
        
        # Parse XML rules (replace with your XML file path)
        xml_file_path = "firewall_rules.xml"  # Update this path
        rules = engine.parse_xml_rules(xml_file_path)
        
        if not rules:
            print("✗ No rules found to import")
            return
        
        # Import rules to database
        success = engine.import_rules_to_database(rules)
        if not success:
            print("✗ Failed to import rules")
            return
        
        # Interactive search loop
        print("\n🚀 Firewall Rule Search Engine Ready!")
        print("Type your search queries in natural language (or 'quit' to exit)")
        
        while True:
            query = input("\n🔍 Search: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not query:
                continue
            
            # Perform search
            results = engine.search_rules(query, limit=5)
            engine.display_search_results(results, query)
    
    except Exception as e:
        print(f"✗ Application error: {e}")


if __name__ == "__main__":
    main()


# Example usage and test queries:
"""
Sample natural language queries you can try:

1. "Show me rules that allow traffic from LAN to WAN"
2. "Find rules with virus scanning enabled"
3. "What rules block malicious traffic?"
4. "Show me default network policies"
5. "Find rules with intrusion prevention"
6. "What rules have zero day protection?"
7. "Show me rules that log traffic"
8. "Find rules for web filtering"
9. "What rules control application access?"
10. "Show me disabled security features"
"""