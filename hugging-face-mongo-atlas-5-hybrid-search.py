import os
import json
import hashlib
import xml.etree.ElementTree as ET
import pymongo
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import re
from collections import defaultdict


@dataclass
class FirewallRule:
    """Enhanced firewall rule with structured indexing"""
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
    # Enhanced indexing fields
    rule_hash: Optional[str] = None
    structured_tags: Optional[List[str]] = None
    exact_match_fields: Optional[Dict[str, Any]] = None
    rule_embedding: Optional[List[float]] = None


class StructuredIndex:
    """Cursor-style structured indexing for precise matches"""
    
    def __init__(self):
        self.field_hashes = defaultdict(set)  # field_name -> set of hashes
        self.hash_to_rules = defaultdict(list)  # hash -> list of rule_ids
        self.tags_index = defaultdict(set)  # tag -> set of rule_ids
    
    def generate_field_hash(self, field_name: str, field_value: Any) -> str:
        """Generate consistent hash for field values"""
        if isinstance(field_value, list):
            # Sort list for consistent hashing
            normalized_value = json.dumps(sorted(field_value), sort_keys=True)
        else:
            normalized_value = str(field_value).lower().strip()
        
        content = f"{field_name}:{normalized_value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def generate_rule_hash(self, rule: FirewallRule) -> str:
        """Generate unique hash for entire rule"""
        rule_content = {
            'name': rule.name,
            'action': rule.action,
            'source_zones': sorted(rule.source_zones),
            'destination_zones': sorted(rule.destination_zones),
            'destination_networks': sorted(rule.destination_networks),
            'status': rule.status
        }
        content = json.dumps(rule_content, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def create_structured_tags(self, rule: FirewallRule) -> List[str]:
        """Create precise tags for exact matching"""
        tags = []
        
        # Zone tags
        for zone in rule.source_zones:
            tags.append(f"src_zone:{zone.lower()}")
        for zone in rule.destination_zones:
            tags.append(f"dst_zone:{zone.lower()}")
        
        # Action tags
        tags.append(f"action:{rule.action.lower()}")
        
        # Status tags
        tags.append(f"status:{rule.status.lower()}")
        
        # Security feature tags
        if rule.virus_scan:
            tags.append("security:virus_scan")
        if rule.zero_day_protection:
            tags.append("security:zero_day")
        if rule.intrusion_prevention and rule.intrusion_prevention != 'None':
            tags.append(f"ips:{rule.intrusion_prevention.lower()}")
        if rule.web_filter and rule.web_filter != 'None':
            tags.append(f"web_filter:{rule.web_filter.lower()}")
        if rule.log_traffic:
            tags.append("logging:enabled")
        
        # Network tags
        for network in rule.destination_networks:
            if network:
                tags.append(f"dst_net:{network.lower()}")
        
        return list(set(tags))  # Remove duplicates


class EnhancedFirewallSearchEngine:
    """Hybrid search engine with structured indexing and semantic search"""
    
    def __init__(self, mongo_url: str, hf_token: str, db_name: str = "firewall_db"):
        self._initialize_connections(mongo_url, hf_token, db_name)
        self._setup_ml_model()
        self.structured_index = StructuredIndex()
        self._setup_search_patterns()
    
    def _initialize_connections(self, mongo_url: str, hf_token: str, db_name: str):
        """Initialize database and authentication connections"""
        self.client = pymongo.MongoClient(mongo_url, tls=True, tlsInsecure=True)
        self.db = self.client[db_name]
        self.firewall_collection = self.db.firewall_rules
        
        if not hf_token or not hf_token.strip():
            raise ValueError("Hugging Face token is required")
        login(token=hf_token)
    
    def _setup_ml_model(self):
        """Load and configure the sentence transformer model"""
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def _setup_search_patterns(self):
        """Define patterns for structured query parsing"""
        self.query_patterns = {
            'source_zone': [
                r'source\s+zone\s*(?:is|=|==)\s*([a-zA-Z0-9_-]+)',
                r'from\s+([a-zA-Z0-9_-]+)\s+(?:zone|to)',
                r'src\s*:\s*([a-zA-Z0-9_-]+)',
            ],
            'destination_zone': [
                r'destination\s+zone\s*(?:is|=|==)\s*([a-zA-Z0-9_-]+)',
                r'to\s+([a-zA-Z0-9_-]+)\s*(?:zone)?',
                r'dst\s*:\s*([a-zA-Z0-9_-]+)',
            ],
            'action': [
                r'action\s*(?:is|=|==)\s*([a-zA-Z]+)',
                r'(allow|deny|drop|reject)\s+traffic',
                r'rules?\s+that\s+(allow|deny|drop|reject)',
            ],
            'status': [
                r'status\s*(?:is|=|==)\s*([a-zA-Z]+)',
                r'(enable|disable)d?\s+rules?',
            ],
            'security_features': [
                r'virus\s+(?:scan|scanning)',
                r'zero\s+day\s+protection',
                r'intrusion\s+prevention',
                r'(?:web\s+)?filter(?:ing)?',
                r'application\s+control',
                r'log(?:ging)?\s+(?:traffic|enabled)',
            ]
        }
    
    def parse_structured_query(self, query: str) -> Dict[str, List[str]]:
        """Parse natural language query for structured elements"""
        structured_filters = defaultdict(list)
        query_lower = query.lower()
        
        # Extract structured filters using patterns
        for field, patterns in self.query_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, query_lower)
                if matches:
                    if field == 'security_features':
                        # Handle security features specially
                        if 'virus' in query_lower:
                            structured_filters['virus_scan'].append('true')
                        if 'zero day' in query_lower:
                            structured_filters['zero_day_protection'].append('true')
                        if 'intrusion' in query_lower:
                            structured_filters['has_ips'].append('true')
                        if any(term in query_lower for term in ['web filter', 'filtering']):
                            structured_filters['has_web_filter'].append('true')
                        if 'log' in query_lower:
                            structured_filters['log_traffic'].append('true')
                    else:
                        structured_filters[field].extend(matches)
        
        return dict(structured_filters)
    
    def build_structured_query(self, structured_filters: Dict[str, List[str]]) -> Dict[str, Any]:
        """Build MongoDB query from structured filters"""
        mongo_query = {}
        
        for field, values in structured_filters.items():
            if field == 'source_zone':
                mongo_query['source_zones'] = {'$in': [v.upper() for v in values]}
            elif field == 'destination_zone':
                mongo_query['destination_zones'] = {'$in': [v.upper() for v in values]}
            elif field == 'action':
                mongo_query['action'] = {'$in': [v.capitalize() for v in values]}
            elif field == 'status':
                mongo_query['status'] = {'$in': [v.capitalize() for v in values]}
            elif field == 'virus_scan':
                mongo_query['virus_scan'] = True
            elif field == 'zero_day_protection':
                mongo_query['zero_day_protection'] = True
            elif field == 'log_traffic':
                mongo_query['log_traffic'] = True
            elif field == 'has_ips':
                mongo_query['intrusion_prevention'] = {'$ne': 'None'}
            elif field == 'has_web_filter':
                mongo_query['web_filter'] = {'$ne': 'None'}
        
        return mongo_query
    
    def parse_xml_rules(self, xml_file_path: str) -> List[FirewallRule]:
        """Parse XML firewall rules into structured objects"""
        # [Previous XML parsing logic remains the same]
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            rules = []
            
            for rule_element in root.findall('FirewallRule'):
                rule = self._extract_rule_data(rule_element)
                if rule:
                    # Add structured indexing
                    rule.rule_hash = self.structured_index.generate_rule_hash(rule)
                    rule.structured_tags = self.structured_index.create_structured_tags(rule)
                    rule.exact_match_fields = {
                        'source_zones': rule.source_zones,
                        'destination_zones': rule.destination_zones,
                        'action': rule.action,
                        'status': rule.status,
                        'virus_scan': rule.virus_scan,
                        'zero_day_protection': rule.zero_day_protection,
                        'log_traffic': rule.log_traffic
                    }
                    rules.append(rule)
            
            print(f"✓ Successfully parsed {len(rules)} firewall rules with structured indexing")
            return rules
            
        except ET.ParseError as e:
            print(f"✗ XML parsing error: {e}")
            return []
        except FileNotFoundError:
            print(f"✗ File not found: {xml_file_path}")
            return []
    
    def _extract_rule_data(self, rule_element: ET.Element) -> Optional[FirewallRule]:
        """Extract and structure firewall rule data"""
        # [Previous extraction logic remains the same]
        try:
            name = self._get_text(rule_element, 'Name')
            description = self._get_text(rule_element, 'Description')
            ip_family = self._get_text(rule_element, 'IPFamily')
            status = self._get_text(rule_element, 'Status')
            position = self._get_text(rule_element, 'Position')
            policy_type = self._get_text(rule_element, 'PolicyType')
            
            network_policy = rule_element.find('NetworkPolicy')
            if network_policy is None:
                return None
            
            action = self._get_text(network_policy, 'Action')
            schedule = self._get_text(network_policy, 'Schedule')
            web_filter = self._get_text(network_policy, 'WebFilter')
            intrusion_prevention = self._get_text(network_policy, 'IntrusionPrevention')
            application_control = self._get_text(network_policy, 'ApplicationControl')
            
            virus_scan = self._get_text(network_policy, 'ScanVirus') == 'Enable'
            zero_day_protection = self._get_text(network_policy, 'ZeroDayProtection') == 'Enable'
            log_traffic = self._get_text(network_policy, 'LogTraffic') == 'Enable'
            
            source_zones = self._extract_list_items(network_policy, 'SourceZones/Zone')
            destination_zones = self._extract_list_items(network_policy, 'DestinationZones/Zone')
            destination_networks = self._extract_list_items(network_policy, 'DestinationNetworks/Network')
            
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
        """Import firewall rules to MongoDB with enhanced indexing"""
        try:
            # Clear existing rules
            self.firewall_collection.delete_many({})
            
            # Process rules with embeddings and structured data
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
                
                # Create indexes for fast exact matching
                self._create_database_indexes()
                
                print(f"✓ Imported {len(result.inserted_ids)} firewall rules with structured indexing")
                return True
            else:
                print("✗ No valid rules to import")
                return False
                
        except Exception as e:
            print(f"✗ Database import error: {e}")
            return False
    
    def _create_database_indexes(self):
        """Create MongoDB indexes for fast structured queries"""
        try:
            # Create indexes for exact matching
            self.firewall_collection.create_index("source_zones")
            self.firewall_collection.create_index("destination_zones")
            self.firewall_collection.create_index("action")
            self.firewall_collection.create_index("status")
            self.firewall_collection.create_index("virus_scan")
            self.firewall_collection.create_index("zero_day_protection")
            self.firewall_collection.create_index("log_traffic")
            self.firewall_collection.create_index("structured_tags")
            self.firewall_collection.create_index("rule_hash")
            
            # Compound indexes for common queries
            self.firewall_collection.create_index([("source_zones", 1), ("action", 1)])
            self.firewall_collection.create_index([("destination_zones", 1), ("status", 1)])
            
            print("✓ Created database indexes for structured search")
        except Exception as e:
            print(f"✗ Error creating indexes: {e}")
    
    def hybrid_search(self, query: str, limit: int = 10) -> Tuple[List[Dict[str, Any]], str]:
        """Hybrid search combining structured and semantic approaches"""
        # Parse query for structured elements
        structured_filters = self.parse_structured_query(query)
        
        search_method = "hybrid"
        
        if structured_filters:
            # Use structured search for precise matches
            mongo_query = self.build_structured_query(structured_filters)
            results = self._structured_search(mongo_query, limit)
            search_method = "structured"
            
            # If structured search yields few results, combine with semantic search
            if len(results) < limit // 2:
                semantic_results = self._semantic_search(query, limit - len(results))
                # Merge results, removing duplicates
                seen_names = {r['name'] for r in results}
                for result in semantic_results:
                    if result['name'] not in seen_names:
                        results.append(result)
                        seen_names.add(result['name'])
                search_method = "hybrid"
        else:
            # Fall back to semantic search for general queries
            results = self._semantic_search(query, limit)
            search_method = "semantic"
        
        return results[:limit], search_method
    
    def _structured_search(self, mongo_query: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Perform structured search using MongoDB queries"""
        try:
            cursor = self.firewall_collection.find(mongo_query).limit(limit)
            results = []
            
            for doc in cursor:
                doc['score'] = 1.0  # Perfect match score
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"✗ Structured search error: {e}")
            return []
    
    def _semantic_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform semantic search using vector embeddings"""
        try:
            query_embedding = self.generate_embedding(query)
            
            if not query_embedding:
                return []
            
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
                        "structured_tags": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            return list(self.firewall_collection.aggregate(pipeline))
            
        except Exception as e:
            print(f"✗ Semantic search error: {e}")
            return []
    
    def search_rules(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Main search interface"""
        results, search_method = self.hybrid_search(query, limit)
        print(f"✓ Found {len(results)} rules using {search_method} search")
        return results
    
    def display_search_results(self, results: List[Dict[str, Any]], query: str):
        """Display search results with search method indication"""
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
            
            # Show structured tags for debugging
            if rule.get('structured_tags'):
                print(f"   Tags: {', '.join(rule['structured_tags'][:5])}{'...' if len(rule['structured_tags']) > 5 else ''}")
            
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
    """Main execution function with enhanced search examples"""
    load_dotenv()
    
    try:
        engine = EnhancedFirewallSearchEngine(
            mongo_url=os.getenv("MONGO_DB_URL"),
            hf_token=os.getenv("HUGGING_FACE_ACCESS_TOKEN")
        )
        
        # Parse XML rules
        xml_file_path = "firewall_rules.xml"
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
        print("\n🚀 Enhanced Firewall Rule Search Engine Ready!")
        print("Now supports both structured and semantic search!")
        print("\nStructured query examples:")
        print("  - 'source zone is LAN'")
        print("  - 'action = Allow'")
        print("  - 'rules with virus scan enabled'")
        print("  - 'from LAN to WAN'")
        print("\nType your search queries (or 'quit' to exit)")
        
        while True:
            query = input("\n🔍 Search: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not query:
                continue
            
            # Perform hybrid search
            results = engine.search_rules(query, limit=5)
            engine.display_search_results(results, query)
    
    except Exception as e:
        print(f"✗ Application error: {e}")


if __name__ == "__main__":
    main()


# Enhanced Example Queries:
"""
Structured queries (exact matches):
1. "source zone is LAN" - Returns only rules with LAN as source
2. "destination zone = WAN" - Returns only rules with WAN as destination  
3. "action Allow" - Returns only Allow rules
4. "status Enable" - Returns only enabled rules
5. "rules with virus scan" - Returns only rules with virus scanning
6. "from LAN to WAN" - Returns rules from LAN to WAN specifically
7. "action Deny and virus scan enabled" - Combined structured filters

Semantic queries (natural language):
8. "Block malicious traffic"
9. "Default security policies"
10. "Web filtering for employees"

The system automatically detects which approach to use!
"""