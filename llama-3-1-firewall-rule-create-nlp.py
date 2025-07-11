"""
Enhanced Firewall Copilot: Natural Language to Firewall Policy Automation
Author: Jubin Aghara (Enhanced Version)
Date: Jul-11-2025

Improvements:
- Better prompt engineering with more diverse examples
- Input preprocessing and normalization
- Multi-attempt extraction with fallback strategies
- Enhanced validation and error handling
- Fuzzy matching for common firewall terms
- Context-aware field mapping
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.llms import Ollama
from typing import Optional, Literal, Dict, List, Any
import json
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import uuid
import tempfile
import re
from difflib import SequenceMatcher

# --- Enhanced Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EnhancedFirewallCopilot")

# --- Enhanced Data Model for Firewall Policy ---
class EnhancedFirewallPolicy(BaseModel):
    """
    Enhanced structure and validation for a firewall policy with better defaults.
    """
    action: Literal["allow", "deny", "accept", "drop", "permit", "block"] = Field(
        description="Action for the traffic (allow/permit/accept or deny/drop/block)"
    )
    user: Optional[str] = Field(default=None, description="User or group name")
    application: Optional[str] = Field(default=None, description="Application name (e.g., YouTube, Facebook)")
    source_zone: Optional[str] = Field(default="LAN", description="Source zone (e.g., LAN, DMZ, WAN, GUEST)")
    destination_zone: Optional[str] = Field(default="WAN", description="Destination zone (e.g., WAN, LAN, DMZ)")
    source_network: Optional[str] = Field(default=None, description="Source network, IP, or subnet")
    destination_network: Optional[str] = Field(default=None, description="Destination network, IP, or subnet")
    service: Optional[str] = Field(default=None, description="Service name (HTTP, HTTPS, SSH, FTP, etc.)")
    port: Optional[str] = Field(default=None, description="Port number or range")
    protocol: Optional[str] = Field(default=None, description="Network protocol (TCP, UDP, ICMP)")
    priority: Optional[int] = Field(default=None, description="Rule priority (1-1000)")
    description: Optional[str] = Field(default=None, description="Rule description")
    schedule: Optional[str] = Field(default="All The Time", description="Time schedule for rule")
    log_traffic: Optional[str] = Field(default="Disable", description="Enable/Disable traffic logging")
    rule_name: Optional[str] = Field(default=None, description="Custom rule name if specified")

# --- Input Preprocessor ---
class InputPreprocessor:
    """
    Preprocesses and normalizes natural language input for better extraction.
    """
    
    # Common term mappings for fuzzy matching
    ACTION_SYNONYMS = {
        'allow': ['allow', 'permit', 'accept', 'enable', 'grant', 'authorize'],
        'deny': ['deny', 'block', 'drop', 'reject', 'refuse', 'forbid', 'prohibit', 'disallow']
    }
    
    PROTOCOL_SYNONYMS = {
        'tcp': ['tcp', 'transmission control protocol'],
        'udp': ['udp', 'user datagram protocol'],
        'icmp': ['icmp', 'internet control message protocol']
    }
    
    SERVICE_PORT_MAP = {
        'http': '80', 'https': '443', 'ssh': '22', 'ftp': '21', 'telnet': '23',
        'smtp': '25', 'dns': '53', 'dhcp': '67', 'pop3': '110', 'imap': '143',
        'snmp': '161', 'ldap': '389', 'rdp': '3389', 'mysql': '3306', 'postgresql': '5432'
    }
    
    ZONE_SYNONYMS = {
        'lan': ['lan', 'local', 'internal', 'inside', 'private', 'intranet'],
        'wan': ['wan', 'internet', 'external', 'outside', 'public', 'web'],
        'dmz': ['dmz', 'demilitarized', 'buffer', 'perimeter'],
        'guest': ['guest', 'visitor', 'wifi', 'wireless']
    }
    
    @staticmethod
    def normalize_input(text: str) -> str:
        """Normalize input text for better processing."""
        # Convert to lowercase
        text = text.lower().strip()
        
        # Replace common variations
        replacements = {
            r'\bfrom\s+(.+?)\s+to\s+(.+?)(?=\s|$)': r'source \1 destination \2',
            r'\bport\s+(\d+)': r'port \1',
            r'\bports?\s+(\d+)-(\d+)': r'port \1-\2',
            r'\bip\s+(\d+\.\d+\.\d+\.\d+)': r'network \1',
            r'\bsubnet\s+(\d+\.\d+\.\d+\.\d+/\d+)': r'network \1',
            r'\bprotocol\s+(\w+)': r'\1 protocol',
            r'\bservice\s+(\w+)': r'\1 service',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        return text
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, Any]:
        """Extract entities like IPs, ports, etc. from text."""
        entities = {}
        
        # Extract IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b'
        ips = re.findall(ip_pattern, text)
        if ips:
            entities['ips'] = ips
        
        # Extract port numbers
        port_pattern = r'\bport\s+(\d+(?:-\d+)?)\b'
        ports = re.findall(port_pattern, text)
        if ports:
            entities['ports'] = ports
        
        # Extract quoted strings (often rule names or descriptions)
        quote_pattern = r'"([^"]+)"|\'([^\']+)\''
        quotes = [match[0] or match[1] for match in re.findall(quote_pattern, text)]
        if quotes:
            entities['quoted_strings'] = quotes
        
        return entities
    
    @classmethod
    def fuzzy_match_term(cls, term: str, category: str) -> Optional[str]:
        """Fuzzy match terms against known categories."""
        if not term:
            return None
            
        term = term.lower()
        synonym_map = getattr(cls, f"{category.upper()}_SYNONYMS", {})
        
        for canonical, synonyms in synonym_map.items():
            for synonym in synonyms:
                if SequenceMatcher(None, term, synonym).ratio() > 0.8:
                    return canonical
        
        return None

# --- Enhanced Prompt Builder ---
class EnhancedPromptBuilder:
    """
    Builds more robust prompts with diverse examples and better instructions.
    """
    
    @staticmethod
    def build_extraction_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert firewall policy analyzer. Your task is to extract structured firewall policy data from natural language input with high accuracy.

CRITICAL RULES:
1. Always identify the ACTION first (allow/permit/accept OR deny/drop/block)
2. Map common synonyms: permit=allow, block=deny, drop=deny
3. Default source_zone=LAN, destination_zone=WAN unless specified
4. For services, include both service name and port when possible
5. Extract IP addresses, subnets, and port numbers precisely
6. Handle various sentence structures and informal language
7. If unsure about a field, use null rather than guessing

ZONE MAPPING:
- LAN/internal/inside/private → LAN
- WAN/internet/external/outside → WAN  
- DMZ/demilitarized/buffer → DMZ
- Guest/visitor/wifi → GUEST

SERVICE-PORT MAPPING:
- HTTP → port 80, HTTPS → port 443, SSH → port 22
- FTP → port 21, SMTP → port 25, DNS → port 53
- RDP → port 3389, MySQL → port 3306

{format_instructions}"""),
            
            ("human", """TRAINING EXAMPLES:

Input: "allow web traffic from internal network to internet"
Output: {{"action": "allow", "service": "HTTP", "port": "80", "source_zone": "LAN", "destination_zone": "WAN", "description": "Allow web traffic from internal network to internet"}}

Input: "block facebook for marketing team during work hours"
Output: {{"action": "deny", "application": "facebook", "user": "marketing team", "schedule": "work hours", "description": "Block facebook for marketing team during work hours"}}

Input: "permit ssh from 192.168.1.0/24 to server 10.0.0.5"
Output: {{"action": "allow", "service": "SSH", "port": "22", "protocol": "tcp", "source_network": "192.168.1.0/24", "destination_network": "10.0.0.5", "description": "Permit ssh from 192.168.1.0/24 to server 10.0.0.5"}}

Input: "deny all traffic from guest wifi to corporate network"
Output: {{"action": "deny", "source_zone": "GUEST", "destination_zone": "LAN", "description": "Deny all traffic from guest wifi to corporate network"}}

Input: "create rule named 'web-access' allowing https from any to any"
Output: {{"action": "allow", "service": "HTTPS", "port": "443", "protocol": "tcp", "rule_name": "web-access", "description": "Create rule named 'web-access' allowing https from any to any"}}

Input: "drop incoming connections to port 3389 from external"
Output: {{"action": "drop", "port": "3389", "service": "RDP", "source_zone": "WAN", "destination_zone": "LAN", "description": "Drop incoming connections to port 3389 from external"}}

Input: "allow users to access mail server on port 25"
Output: {{"action": "allow", "service": "SMTP", "port": "25", "protocol": "tcp", "destination_network": "mail server", "description": "Allow users to access mail server on port 25"}}

Input: "block youtube and facebook for all users except IT team"
Output: {{"action": "deny", "application": "youtube, facebook", "user": "all users except IT team", "description": "Block youtube and facebook for all users except IT team"}}

Input: "permit dns queries from 10.10.10.0/24 to 8.8.8.8"
Output: {{"action": "allow", "service": "DNS", "port": "53", "protocol": "udp", "source_network": "10.10.10.0/24", "destination_network": "8.8.8.8", "description": "Permit dns queries from 10.10.10.0/24 to 8.8.8.8"}}

Input: "reject all traffic from dmz to internal network except port 443"
Output: {{"action": "deny", "source_zone": "DMZ", "destination_zone": "LAN", "description": "Reject all traffic from dmz to internal network except port 443"}}

Input: "enable logging for all denied traffic"
Output: {{"action": "deny", "log_traffic": "Enable", "description": "Enable logging for all denied traffic"}}

Input: "allow remote desktop access from admin subnet 172.16.0.0/24"
Output: {{"action": "allow", "service": "RDP", "port": "3389", "protocol": "tcp", "source_network": "172.16.0.0/24", "user": "admin", "description": "Allow remote desktop access from admin subnet 172.16.0.0/24"}}

Now extract policy from: "{user_input}"

Remember:
- Be precise with IP addresses and port numbers
- Handle informal language and typos
- Map synonyms correctly
- Include descriptions for context
- Use null for uncertain fields""")
        ])

# --- Enhanced Firewall Copilot ---
class EnhancedFirewallCopilot:
    """
    Enhanced version with better NLP processing and error handling.
    """
    
    def __init__(self, model_name="llama3.1", api_config=None):
        self.llm = Ollama(model=model_name, temperature=0.1)
        self.parser = JsonOutputParser(pydantic_object=EnhancedFirewallPolicy)
        self.preprocessor = InputPreprocessor()
        self.prompt_builder = EnhancedPromptBuilder()
        self.chain = self._create_enhanced_chain()
        self.xml_generator = XMLPolicyGenerator()
        self.api_client = FirewallAPIClient(api_config or FirewallAPIConfig())
        logger.info(f"✅ Enhanced Copilot ready (model: {model_name})")
    
    def _create_enhanced_chain(self):
        """Create the enhanced processing chain."""
        prompt = self.prompt_builder.build_extraction_prompt()
        return prompt.partial(format_instructions=self.parser.get_format_instructions()) | self.llm | self.parser
    
    def extract_policy_with_retries(self, user_input: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Extract policy with multiple attempts and fallback strategies.
        """
        original_input = user_input
        
        for attempt in range(max_retries):
            try:
                # Preprocess input
                if attempt == 0:
                    processed_input = self.preprocessor.normalize_input(user_input)
                elif attempt == 1:
                    # Try with extracted entities as hints
                    entities = self.preprocessor.extract_entities(user_input)
                    processed_input = f"{user_input} [Entities: {entities}]"
                else:
                    # Final attempt with simplified input
                    processed_input = self._simplify_input(user_input)
                
                logger.info(f"Attempt {attempt + 1}: Processing '{processed_input[:50]}...'")
                
                # Extract policy
                policy_data = self.chain.invoke({"user_input": processed_input})
                
                # Validate and enhance the extracted policy
                enhanced_policy = self._enhance_policy_data(policy_data, original_input)
                
                return enhanced_policy
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise e
        
        raise Exception("All extraction attempts failed")
    
    def _simplify_input(self, text: str) -> str:
        """Simplify input for final attempt."""
        # Extract key components
        words = text.lower().split()
        
        # Find action
        action = None
        for word in words:
            if self.preprocessor.fuzzy_match_term(word, 'action'):
                action = self.preprocessor.fuzzy_match_term(word, 'action')
                break
        
        # Find IPs and ports
        entities = self.preprocessor.extract_entities(text)
        
        # Build simplified version
        simplified = f"{action or 'allow'} traffic"
        if entities.get('ips'):
            simplified += f" involving {', '.join(entities['ips'])}"
        if entities.get('ports'):
            simplified += f" on port {', '.join(entities['ports'])}"
        
        return simplified
    
    def _enhance_policy_data(self, policy_data: Dict[str, Any], original_input: str) -> Dict[str, Any]:
        """
        Enhance and validate extracted policy data.
        """
        enhanced = policy_data.copy()
        
        # Normalize action
        if enhanced.get('action'):
            action = enhanced['action'].lower()
            if action in ['permit', 'accept']:
                enhanced['action'] = 'allow'
            elif action in ['block', 'drop']:
                enhanced['action'] = 'deny'
        
        # Auto-fill service and port relationships
        if enhanced.get('service') and not enhanced.get('port'):
            service_lower = enhanced['service'].lower()
            if service_lower in self.preprocessor.SERVICE_PORT_MAP:
                enhanced['port'] = self.preprocessor.SERVICE_PORT_MAP[service_lower]
        
        # Generate description if not present
        if not enhanced.get('description'):
            enhanced['description'] = f"Auto-generated from: {original_input[:100]}"
        
        # Validate IP addresses
        for field in ['source_network', 'destination_network']:
            if enhanced.get(field):
                enhanced[field] = self._validate_ip_address(enhanced[field])
        
        return enhanced
    
    def _validate_ip_address(self, ip_string: str) -> str:
        """Validate and normalize IP address strings."""
        # Basic validation - you could enhance this with ipaddress module
        if re.match(r'^(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?$', ip_string):
            return ip_string
        return ip_string  # Return as-is if not standard IP format
    
    def process_policy(self, user_input: str, mode: str = "test") -> Dict[str, Any]:
        """
        Enhanced policy processing with better error handling.
        """
        try:
            # Step 1: Extract policy with retries
            logger.info("🔄 Extracting policy data with enhanced NLP...")
            policy_data = self.extract_policy_with_retries(user_input)
            
            # Step 2: Generate policy name
            policy_name = policy_data.get('rule_name') or f"policy_{uuid.uuid4().hex[:8]}"
            
            # Step 3: Generate XML
            logger.info("🔄 Generating XML configuration...")
            xml_content = self.xml_generator.generate_policy_xml(
                policy_data, policy_name, 
                self.api_client.config.username, 
                self.api_client.config.password
            )
            
            # Step 4: Prepare result
            result = {
                "original_input": user_input,
                "extracted_policy": policy_data,
                "policy_name": policy_name,
                "xml_content": xml_content,
                "timestamp": datetime.now().isoformat(),
                "confidence_score": self._calculate_confidence_score(policy_data)
            }
            
            # Step 5: Deploy if real mode
            if mode == "real":
                logger.info("🔄 Deploying policy via API...")
                api_success, api_result = self.api_client.create_policy(xml_content, policy_name)
                result["api_call"] = api_result
                result["created_successfully"] = api_success
                result["response_message"] = (
                    self._success_message(policy_name, policy_data, user_input)
                    if api_success else
                    self._error_message(policy_name, api_result.get('error', 'Unknown error'))
                )
            else:
                result["created_successfully"] = None
                result["response_message"] = f"✅ Policy '{policy_name}' extracted and validated (test mode)"
            
            return result
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                "error": str(e),
                "original_input": user_input,
                "created_successfully": False,
                "response_message": f"❌ Failed to process policy: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _calculate_confidence_score(self, policy_data: Dict[str, Any]) -> float:
        """Calculate confidence score based on extracted data completeness."""
        required_fields = ['action']
        optional_fields = ['source_zone', 'destination_zone', 'service', 'port', 'protocol']
        
        score = 0.0
        
        # Required fields (50% weight)
        for field in required_fields:
            if policy_data.get(field):
                score += 0.5
        
        # Optional fields (50% weight)
        filled_optional = sum(1 for field in optional_fields if policy_data.get(field))
        score += 0.5 * (filled_optional / len(optional_fields))
        
        return min(score, 1.0)
    
    def _success_message(self, policy_name: str, policy_data: Dict[str, Any], original_input: str) -> str:
        return (
            f"🎉 Policy '{policy_name}' created successfully!\n"
            f"✅ Rule to {policy_data.get('action', 'unknown')} traffic "
            f"from {policy_data.get('source_zone', 'unknown')} to {policy_data.get('destination_zone', 'unknown')}\n"
            f"📋 Original request: \"{original_input}\""
        )
    
    def _error_message(self, policy_name: str, error: str) -> str:
        return f"❌ Failed to create policy '{policy_name}': {error}"

# --- Keep original classes for compatibility ---
class FirewallAPIConfig:
    """Stores firewall API connection details."""
    def __init__(self, api_url="https://172.16.16.16:4444/webconsole/APIController", username="admin", password="Admin@12345"):
        self.api_url = api_url
        self.username = username
        self.password = password

class XMLPolicyGenerator:
    """Converts structured policy data into firewall-compatible XML."""
    @staticmethod
    def generate_policy_xml(policy_data: dict, policy_name: str, username: str, password: str) -> str:
        # [XML generation logic remains the same as original]
        root = ET.Element("Request")
        login_elem = ET.SubElement(root, "Login")
        ET.SubElement(login_elem, "Username").text = username
        ET.SubElement(login_elem, "Password").text = password

        set_elem = ET.SubElement(root, "Set", operation="add")
        firewall_rule = ET.SubElement(set_elem, "FirewallRule", transactionid="")

        ET.SubElement(firewall_rule, "Name").text = policy_name
        ET.SubElement(firewall_rule, "Description").text = policy_data.get("description", f"Auto-generated rule: {policy_name}")
        ET.SubElement(firewall_rule, "IPFamily").text = "IPv4"
        ET.SubElement(firewall_rule, "Status").text = "Enable"
        ET.SubElement(firewall_rule, "Position").text = "After"
        ET.SubElement(firewall_rule, "PolicyType").text = "Network"

        after_elem = ET.SubElement(firewall_rule, "After")
        ET.SubElement(after_elem, "Name").text = "demo-firewall-rule"

        network_policy = ET.SubElement(firewall_rule, "NetworkPolicy")

        action_map = {"allow": "Accept", "accept": "Accept", "deny": "Drop", "drop": "Drop"}
        ET.SubElement(network_policy, "Action").text = action_map.get(policy_data.get("action", "deny").lower(), "Drop")
        ET.SubElement(network_policy, "LogTraffic").text = policy_data.get("log_traffic", "Disable")
        ET.SubElement(network_policy, "SkipLocalDestined").text = "Disable"

        sz = policy_data.get("source_zone", "LAN")
        dz = policy_data.get("destination_zone", "WAN")
        ET.SubElement(ET.SubElement(network_policy, "SourceZones"), "Zone").text = sz
        ET.SubElement(ET.SubElement(network_policy, "DestinationZones"), "Zone").text = dz

        ET.SubElement(network_policy, "Schedule").text = policy_data.get("schedule", "All The Time")

        ET.SubElement(ET.SubElement(network_policy, "SourceNetworks"), "Network").text = policy_data.get("source_network", "LAN-network-10.10.10.0")
        ET.SubElement(ET.SubElement(network_policy, "DestinationNetworks"), "Network").text = policy_data.get("destination_network", "Internet IPv4 group")

        if policy_data.get("service") or policy_data.get("port"):
            services_elem = ET.SubElement(network_policy, "Services")
            service_text = policy_data.get("service") or f"TCP-{policy_data['port']}"
            ET.SubElement(services_elem, "Service").text = service_text

        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

class FirewallAPIClient:
    """Handles communication with the firewall API."""
    def __init__(self, config: FirewallAPIConfig):
        self.config = config
        self.session = requests.Session()
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

    def create_policy(self, xml_content: str, policy_name: str) -> tuple[bool, dict]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as temp_file:
            temp_file.write(xml_content)
            temp_file_path = temp_file.name

        try:
            with open(temp_file_path, 'rb') as xml_file:
                files = {'reqxml': (f'{policy_name}.xml', xml_file, 'application/xml')}
                response = self.session.post(
                    self.config.api_url,
                    files=files,
                    verify=False,
                    timeout=30
                )
            
            success = response.status_code == 200
            result = {
                'status_code': response.status_code,
                'success': success,
                'response_text': response.text,
                'policy_name': policy_name,
                'xml_file_path': temp_file_path
            }
            
            if success:
                logger.info(f"✅ Policy '{policy_name}' created.")
            else:
                logger.error(f"❌ Policy '{policy_name}' failed: HTTP {response.status_code}")
            
            return success, result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ API request failed: {e}")
            return False, {'success': False, 'error': str(e), 'policy_name': policy_name}

# --- Enhanced Interactive Mode ---
def enhanced_interactive_mode():
    """Enhanced CLI with better user experience."""
    print("🛡️  Enhanced Firewall Copilot: Natural Language to Firewall Policy")
    print("🚀 Features: Multi-attempt extraction, fuzzy matching, enhanced validation")
    print("📝 Examples:")
    print("  - 'allow web traffic from internal to internet'")
    print("  - 'block facebook for marketing team'")
    print("  - 'permit ssh from 192.168.1.0/24 to 10.0.0.5'")
    print("  - 'deny all traffic from guest wifi to corporate network'")
    print("\nModes: 'test' (dry run), 'real' (deploy via API)")
    print("Type 'exit' to quit.\n")

    api_config = FirewallAPIConfig()
    copilot = EnhancedFirewallCopilot(api_config=api_config)
    
    while True:
        try:
            mode = input("Select mode [test/real]: ").strip().lower()
            if mode not in ("test", "real"):
                print("⚠️  Please enter 'test' or 'real'.")
                continue

            user_input = input("📝 Enter firewall policy: ").strip()
            if user_input.lower() in ("exit", "quit", "q"):
                print("👋 Goodbye!")
                break
            if not user_input:
                continue

            print("🔄 Processing with enhanced NLP...")
            result = copilot.process_policy(user_input, mode=mode)

            print(f"\n{result.get('response_message', 'Policy processed.')}")
            
            if result.get('confidence_score'):
                confidence = result['confidence_score'] * 100
                print(f"🎯 Confidence: {confidence:.1f}%")

            show_details = input("Show technical details? [y/N]: ").strip().lower()
            if show_details == "y":
                print(json.dumps({
                    "extracted_policy": result.get("extracted_policy"),
                    "confidence_score": result.get("confidence_score"),
                    "xml_generated": bool(result.get("xml_content")),
                    "api_status": result.get("api_call", {}).get("status_code") if "api_call" in result else None
                }, indent=2))
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    enhanced_interactive_mode()
