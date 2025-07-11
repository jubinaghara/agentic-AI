"""
Firewall Copilot: Natural Language to Firewall Policy Automation
Author: Jubin Aghara
Date: Jul-11-2025
Updated: Azure OpenAI Integration

Features:
- Accepts natural language firewall policy requests
- Extracts structured policy data using LLM (LangChain + Azure OpenAI)
- Generates XML config for firewall
- Supports two modes: test (dry run) and real (deploy via API)
"""

import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from typing import Optional, Literal
import json
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import uuid
import tempfile

# Load environment variables
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FirewallCopilot")

# --- Data Model for Firewall Policy ---
class FirewallPolicy(BaseModel):
    """
    Defines the structure and validation for a firewall policy.
    """
    action: Literal["allow", "deny", "accept", "drop"] = Field(description="Action for the traffic")
    user: Optional[str] = Field(default=None, description="User or group name")
    application: Optional[str] = Field(default=None, description="Application name (e.g., YouTube)")
    source_zone: Optional[str] = Field(default="LAN", description="Source zone (e.g., LAN, DMZ)")
    destination_zone: Optional[str] = Field(default="WAN", description="Destination zone (e.g., WAN, LAN)")
    source_network: Optional[str] = Field(default=None, description="Source network or IP range")
    destination_network: Optional[str] = Field(default=None, description="Destination network or IP range")
    service: Optional[str] = Field(default=None, description="Service or protocol (HTTP, HTTPS, SSH)")
    port: Optional[str] = Field(default=None, description="Port number")
    protocol: Optional[str] = Field(default=None, description="Network protocol (TCP, UDP)")
    priority: Optional[int] = Field(default=None, description="Rule priority")
    description: Optional[str] = Field(default=None, description="Rule description")
    schedule: Optional[str] = Field(default="All The Time", description="Rule schedule")
    log_traffic: Optional[str] = Field(default="Disable", description="Enable/Disable traffic logging")

# --- Azure OpenAI Configuration ---
class AzureOpenAIConfig:
    """
    Azure OpenAI configuration and validation.
    """
    def __init__(self):
        # Get credentials from environment variables
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # Validate required environment variables
        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise ValueError(
                "Missing required Azure OpenAI environment variables. "
                "Please set: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_DEPLOYMENT_NAME in your .env file"
            )
        
        # Validate endpoint format
        if not self.endpoint.startswith(('https://', 'http://')):
            raise ValueError(f"Invalid endpoint format: {self.endpoint}. Must start with https:// or http://")
        
        logger.info("✅ Azure OpenAI configuration validated")
        logger.info(f"📋 Endpoint: {self.endpoint}")
        logger.info(f"📋 Deployment: {self.deployment_name}")
        logger.info(f"📋 API Version: {self.api_version}")

# --- API Connection Settings ---
class FirewallAPIConfig:
    """
    Stores firewall API connection details.
    """
    def __init__(self, api_url="https://172.16.16.16:4444/webconsole/APIController", username="admin", password="Admin@12345"):
        self.api_url = api_url
        self.username = username
        self.password = password

# --- XML Generator for Firewall Policy ---
class XMLPolicyGenerator:
    """
    Converts structured policy data into firewall-compatible XML.
    """
    @staticmethod
    def generate_policy_xml(policy_data: dict, policy_name: str, username: str, password: str) -> str:
        # Build the XML tree for the firewall policy.
        root = ET.Element("Request")
        login_elem = ET.SubElement(root, "Login")
        ET.SubElement(login_elem, "Username").text = username
        ET.SubElement(login_elem, "Password").text = password

        set_elem = ET.SubElement(root, "Set", operation="add")
        firewall_rule = ET.SubElement(set_elem, "FirewallRule", transactionid="")

        # Rule name and description
        ET.SubElement(firewall_rule, "Name").text = policy_name
        ET.SubElement(firewall_rule, "Description").text = policy_data.get("description", f"Auto-generated rule: {policy_name}")
        ET.SubElement(firewall_rule, "IPFamily").text = "IPv4"
        ET.SubElement(firewall_rule, "Status").text = "Enable"
        ET.SubElement(firewall_rule, "Position").text = "After"
        ET.SubElement(firewall_rule, "PolicyType").text = "Network"

        # Position reference (after a default rule)
        after_elem = ET.SubElement(firewall_rule, "After")
        ET.SubElement(after_elem, "Name").text = "demo-firewall-rule"

        # Network policy section
        network_policy = ET.SubElement(firewall_rule, "NetworkPolicy")

        # Map action to firewall terminology
        action_map = {"allow": "Accept", "accept": "Accept", "deny": "Drop", "drop": "Drop"}
        ET.SubElement(network_policy, "Action").text = action_map.get(policy_data.get("action", "deny").lower(), "Drop")
        ET.SubElement(network_policy, "LogTraffic").text = policy_data.get("log_traffic", "Disable")
        ET.SubElement(network_policy, "SkipLocalDestined").text = "Disable"

        # Source and destination zones
        sz = policy_data.get("source_zone", "LAN")
        dz = policy_data.get("destination_zone", "WAN")
        ET.SubElement(ET.SubElement(network_policy, "SourceZones"), "Zone").text = sz
        ET.SubElement(ET.SubElement(network_policy, "DestinationZones"), "Zone").text = dz

        # Schedule
        ET.SubElement(network_policy, "Schedule").text = policy_data.get("schedule", "All The Time")

        # Networks
        ET.SubElement(ET.SubElement(network_policy, "SourceNetworks"), "Network").text = policy_data.get("source_network", "LAN-network-10.10.10.0")
        ET.SubElement(ET.SubElement(network_policy, "DestinationNetworks"), "Network").text = policy_data.get("destination_network", "Internet IPv4 group")

        # Services (if specified)
        if policy_data.get("service") or policy_data.get("port"):
            services_elem = ET.SubElement(network_policy, "Services")
            service_text = policy_data.get("service") or f"TCP-{policy_data['port']}"
            ET.SubElement(services_elem, "Service").text = service_text

        # Return formatted XML string
        xml_str = ET.tostring(root, encoding='unicode', method='xml')
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str

# --- API Client for Firewall ---
class FirewallAPIClient:
    """
    Handles communication with the firewall API.
    """
    def __init__(self, config: FirewallAPIConfig):
        self.config = config
        self.session = requests.Session()
        requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

    def create_policy(self, xml_content: str, policy_name: str) -> tuple[bool, dict]:
        """
        Sends the XML policy to the firewall API.
        Returns (success, result_dict).
        """
        # Write XML to a temporary file for upload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as temp_file:
            temp_file.write(xml_content)
            temp_file_path = temp_file.name

        try:
            with open(temp_file_path, 'rb') as xml_file:
                files = {'reqxml': (f'{policy_name}.xml', xml_file, 'application/xml')}
                response = self.session.post(
                    self.config.api_url,
                    files=files,
                    verify=False,  # Ignore SSL
                    timeout=30
                )
            # Prepare result
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

# --- Main Copilot Class ---
class FirewallCopilot:
    """
    Orchestrates NLP extraction, XML generation, and API operations using Azure OpenAI.
    """
    def __init__(self, azure_config=None, api_config=None):
        # Initialize Azure OpenAI configuration
        self.azure_config = azure_config or AzureOpenAIConfig()
        
        # Initialize Azure OpenAI LLM with better error handling
        try:
            self.llm = AzureChatOpenAI(
                azure_endpoint=self.azure_config.endpoint,
                api_key=self.azure_config.api_key,
                api_version=self.azure_config.api_version,
                azure_deployment=self.azure_config.deployment_name,
                temperature=0.1,
                max_tokens=1500,
                timeout=30,  # 30 second timeout
                max_retries=2  # Limit retries
            )
            
            # Test the connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI: {e}")
            raise ValueError(f"Azure OpenAI initialization failed: {e}")
        
        # Initialize parser, chain, and other components
        self.parser = JsonOutputParser(pydantic_object=FirewallPolicy)
        self.chain = self._create_chain()
        self.xml_generator = XMLPolicyGenerator()
        self.api_client = FirewallAPIClient(api_config or FirewallAPIConfig())
        
        logger.info(f"✅ Firewall Copilot ready (Azure OpenAI: {self.azure_config.deployment_name})")

    def _test_connection(self):
        """Test Azure OpenAI connection with a simple query"""
        try:
            logger.info("🔄 Testing Azure OpenAI connection...")
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ]
            
            # Use the raw client for testing
            from langchain_core.messages import HumanMessage, SystemMessage
            test_response = self.llm.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Hello")
            ])
            
            logger.info("✅ Azure OpenAI connection successful")
            
        except Exception as e:
            logger.error(f"❌ Azure OpenAI connection test failed: {e}")
            raise ConnectionError(f"Cannot connect to Azure OpenAI: {e}")

    def _create_chain(self):
        # Build the AI prompt chain for extracting policy data
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert firewall policy analyzer. Extract structured firewall policy data from natural language input.

Guidelines:
- Identify action (allow/deny/accept/drop)
- Extract zones, networks, services, ports, users, applications
- Use firewall terminology
- Default: source zone=LAN, destination zone=WAN
- Return valid JSON only, no additional text

{format_instructions}"""),
            ("human", """Examples:
     
Input: "create firewall rule name demo-2 to allow access to destination network 1.2.2.3 from source ip 1.1.1.1. source zone is LAN and destination zone is WAN"
Output: {{"action": "allow", "name": "demo-2", "source_network": "1.1.1.1", "destination_network": "1.2.2.3", "source_zone": "LAN", "destination_zone": "WAN"}}
     
Input: "allow user from source network 1.1.1.1 to destination network 2.2.2.2"
Output: {{"action": "allow", "source_network": "1.1.1.1", "destination_network": "2.2.2.2"}}

Input: "allow all traffic from LAN to WAN"
Output: {{"action": "allow", "source_zone": "LAN", "destination_zone": "WAN"}}

Input: "deny ssh access from 192.168.1.0/24 to 10.0.0.5"
Output: {{"action": "deny", "service": "SSH", "protocol": "tcp", "source_network": "192.168.1.0/24", "destination_network": "10.0.0.5", "port": "22"}}

Input: "block youtube for marketing team during office hours"
Output: {{"action": "deny", "application": "youtube", "user": "marketing team", "schedule": "office hours"}}

Input: "permit https from dmz to internet"
Output: {{"action": "allow", "source_zone": "DMZ", "destination_zone": "WAN", "service": "HTTPS", "port": "443", "protocol": "tcp"}}

Input: "drop all incoming traffic to port 3389"
Output: {{"action": "drop", "destination_zone": "LAN", "port": "3389", "service": "RDP"}}

Input: "accept tcp traffic from guest wifi to 172.16.0.0/16 on port 80"
Output: {{"action": "accept", "source_zone": "GUEST", "destination_network": "172.16.0.0/16", "service": "HTTP", "port": "80", "protocol": "tcp"}}

Input: "allow DNS queries from any to 8.8.8.8"
Output: {{"action": "allow", "service": "DNS", "port": "53", "protocol": "udp", "destination_network": "8.8.8.8"}}

Input: "deny all traffic from external to internal"
Output: {{"action": "deny", "source_zone": "WAN", "destination_zone": "LAN"}}

Input: "allow smtp from 10.10.10.5 to mail server"
Output: {{"action": "allow", "service": "SMTP", "port": "25", "protocol": "tcp", "source_network": "10.10.10.5", "destination_network": "mail server"}}

Input: "block facebook for all users"
Output: {{"action": "deny", "application": "facebook"}}

Input: "allow users from internal network to access internet over https"
Output: {{"action": "allow", "source_zone": "LAN", "destination_zone": "WAN", "source_network": "LAN-network-10.10.10.0", "destination_network": "Internet IPv4 group", "service": "HTTPS", "port": "443", "protocol": "tcp"}}

Input: "deny all traffic from dmz to lan on port 22"
Output: {{"action": "deny", "source_zone": "DMZ", "destination_zone": "LAN", "service": "SSH", "port": "22", "protocol": "tcp"}}

Now extract policy from this input: "{user_input}" """)
        ])
        return prompt.partial(format_instructions=self.parser.get_format_instructions()) | self.llm | self.parser

    def process_policy(self, user_input: str, mode: str = "test") -> dict:
        """
        Processes user input; extracts policy, generates XML, and (optionally) deploys.
        mode: "test" (dry run) or "real" (deploy via API)
        """
        try:
            # Step 1: Extract structured policy from input using Azure OpenAI
            logger.info("🔄 Extracting policy data with Azure OpenAI...")
            
            try:
                policy_data = self.chain.invoke({"user_input": user_input})
                logger.info("✅ Policy extraction successful")
                
            except Exception as llm_error:
                logger.error(f"❌ LLM processing failed: {llm_error}")
                # Check if it's a connection issue
                if "Connection error" in str(llm_error) or "timeout" in str(llm_error).lower():
                    error_msg = (
                        "Connection to Azure OpenAI failed. Please check:\n"
                        "1. Your internet connection\n"
                        "2. Azure OpenAI endpoint URL is correct\n"
                        "3. API key is valid\n"
                        "4. Deployment name exists\n"
                        "5. Azure OpenAI service is running"
                    )
                elif "authentication" in str(llm_error).lower() or "unauthorized" in str(llm_error).lower():
                    error_msg = (
                        "Authentication failed. Please check:\n"
                        "1. API key is correct\n"
                        "2. API key has proper permissions\n"
                        "3. Deployment name is correct"
                    )
                elif "not found" in str(llm_error).lower():
                    error_msg = (
                        "Resource not found. Please check:\n"
                        "1. Deployment name is correct\n"
                        "2. Endpoint URL is correct\n"
                        "3. API version is supported"
                    )
                else:
                    error_msg = f"Azure OpenAI error: {llm_error}"
                
                return {
                    "error": str(llm_error),
                    "original_input": user_input,
                    "created_successfully": False,
                    "response_message": error_msg,
                    "timestamp": datetime.now().isoformat()
                }

            # Step 2: Generate unique policy name
            policy_name = f"policy_{uuid.uuid4().hex[:8]}"

            # Step 3: Generate XML configuration
            logger.info("🔄 Generating XML...")
            xml_content = self.xml_generator.generate_policy_xml(
                policy_data, policy_name, self.api_client.config.username, self.api_client.config.password
            )

            result = {
                "original_input": user_input,
                "extracted_policy": policy_data,
                "policy_name": policy_name,
                "xml_content": xml_content,
                "timestamp": datetime.now().isoformat()
            }

            # Step 4: Deploy via API if mode is "real"
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
                result["response_message"] = f"Policy '{policy_name}' extracted and XML generated (test mode, no API call)."

            return result

        except Exception as e:
            logger.error(f"Processing error: {e}")
            return {
                "error": str(e),
                "original_input": user_input,
                "created_successfully": False,
                "response_message": f"Failed to process policy: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _success_message(self, policy_name, policy_data, original_input):
        # Friendly message for successful deployment
        return (
            f"🎉 Policy '{policy_name}' created!\n"
            f"✅ Rule to {policy_data.get('action', 'unknown')} traffic from {policy_data.get('source_zone', 'unknown')} "
            f"to {policy_data.get('destination_zone', 'unknown')} is now active.\n"
            f"📋 Your request: \"{original_input}\" has been implemented."
        )

    def _error_message(self, policy_name, error):
        # Friendly message for errors
        return f"❌ Could not create policy '{policy_name}'. Error: {error}"

# --- Interactive CLI ---
def interactive_mode():
    """
    Command-line interface for Firewall Copilot.
    Modes: test (dry run), real (deploy via API).
    """
    print("🛡️  Firewall Copilot: Natural Language to Firewall Policy (Azure OpenAI)")
    print("Modes: 'test' (dry run, no API call), 'real' (deploy via API)")
    print("Type 'exit' to quit.\n")

    try:
        # Initialize with Azure OpenAI and default API config
        azure_config = AzureOpenAIConfig()
        api_config = FirewallAPIConfig()
        copilot = FirewallCopilot(azure_config=azure_config, api_config=api_config)
        all_results = []

        while True:
            try:
                # Prompt for mode
                mode = input("Select mode [test/real]: ").strip().lower()
                if mode not in ("test", "real"):
                    print("Please enter 'test' or 'real'.")
                    continue

                # Prompt for policy input
                user_input = input("📝 Enter firewall policy (natural language): ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    print("👋 Exiting Firewall Copilot.")
                    break
                if not user_input:
                    continue

                # Process the policy
                print("🔄 Processing with Azure OpenAI...")
                result = copilot.process_policy(user_input, mode=mode)
                all_results.append(result)

                # Show result message
                print("\n" + result.get("response_message", "Policy processed."))

                # Optionally show technical details
                show_details = input("Show technical details? [y/N]: ").strip().lower()
                if show_details == "y":
                    print(json.dumps({
                        "extracted_policy": result.get("extracted_policy"),
                        "xml_generated": bool(result.get("xml_content")),
                        "api_status": result.get("api_call", {}).get("status_code") if "api_call" in result else None
                    }, indent=2))
                print()  # Blank line for readability

            except KeyboardInterrupt:
                print("\n👋 Exiting Firewall Copilot.")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                print(f"❌ Error: {e}\n")

    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nPlease create a .env file with the following variables:")
        print("AZURE_OPENAI_API_KEY=your_api_key")
        print("AZURE_OPENAI_ENDPOINT=your_endpoint")
        print("AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name")
        print("AZURE_OPENAI_API_VERSION=2024-02-01  # Optional, defaults to 2024-02-01")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")

if __name__ == "__main__":
    interactive_mode()