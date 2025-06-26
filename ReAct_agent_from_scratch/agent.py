from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
import json 

# Utility function to read content from a file
def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()

# Wrapper class for tools/actions that the agent can use
class Tool:
    def __init__(self, name, func):
        self.name = name      # Name of the tool (e.g., "multiply")
        self.func = func      # Function reference that implements the tool

    # Executes the tool's function with given arguments
    def execute(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
# Main agent class responsible for conversation, reasoning, and tool execution
# TODO: Yet to utilize the thoughts, actions and reasons data.
class Agent:
    def __init__(self, max_iterations=5):
        self.tools: Dict[str, Tool] = {}    # Registered tools keyed by name
        self.iterations = 0                 # Tracks how many reasoning cycles completed
        self.max_iterations = max_iterations  # Max allowed reasoning cycles before stopping
        self.messages = []                  # Stores chat messages exchanged
        self.thoughts = []                  # Stores internal thoughts (if needed)
        self.actions = []                   # Records actions decided by agent
        self.reasons = []                   # Reasons behind each action
        self.observations = []              # Results returned by tools after actions
        self.final_answer = None            # Stores the final answer when found
        self.log = []                      # Optionally, detailed log of steps
        self.intermediate_steps = []       # Stores any intermediate reasoning results
        self.stop = False                  # Flag to indicate when to stop iterating
        self.query = ""                   # User's original query
        self.model = None                  # Language model instance to be used
        self.load_model("anthropic.claude-3-5-sonnet-20240620-v1:0")  # Load specific model
    
    # Loads a prompt template from a file and stores it for reuse
    def load_template(self, filename):
        self.template = ChatPromptTemplate.from_template(read_file(filename))
    
    # Initializes the language model with a given name/provider
    def load_model(self, model_name):
        self.model = init_chat_model(model_name, model_provider="bedrock_converse")

    # Registers a new tool by associating a name with a callable function
    def register_tool(self, name, func):
        self.tools[name] = Tool(name, func)

    # Adds a new message to the conversation history with a role and content
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    # Retrieves the conversation history as a formatted string
    def get_history(self):
        return "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in self.messages])
    
    # Constructs the prompt for the language model using the template and current state
    def get_prompt(self):
        prompt = self.template.invoke({
          "query": self.query,                # Current user query
          "history": self.get_history(),     # Past messages to provide context
          "tools": list(self.tools.keys())   # List of available tool names
        })
        return prompt
        
    # Sends the prompt to the language model and returns the model's response
    def invoke_model(self, prompt):
        response = self.model.invoke(prompt)
        # print("Model response:", response)
        return response
        
    # Core reasoning loop - makes the agent "think" and decide next steps
    def think(self):
        self.iterations += 1
        # Stop if max iterations exceeded - avoid infinite loops
        if (self.iterations > self.max_iterations):
          self.messages.append(
            { "role": "system", "content": "Sorry I was unable to get the specific answer in certain iterations. Here is the information i gathered till now." }
          )
          return
        # Stop if already reached a final answer
        if self.stop:
          return self.final_answer      
        # Build prompt, invoke model, and process response
        prompt = self.get_prompt()
        response = self.invoke_model(prompt=prompt)
        print("response in think: ", response)
        self.decide(response)
        # return response
    
    # Processes the model's response to decide next action or conclude
    def decide(self, response):
        parsed_json = json.loads(response.content)   # Expect response in JSON format
        if "answer" in parsed_json:
            # If an answer is provided, save it and mark done
            self.final_answer = parsed_json.get("answer")
            self.messages.append({"role": "system", "content": self.final_answer})
            self.stop = True
            # return self.final_answer
        elif "action" in parsed_json:
          # If an action is requested, proceed to execute it
          print("response content in decide: ", response.content)
          self.act(parsed_json.get("action"))
          
    # Executes a specified action by calling the registered tool with inputs
    def act(self, action):
      print("calling act with action: ", action)
      action_name, action_input, action_reason = action["name"], action["input"], action["reason"]
      
      if action_name in self.tools:
        # Handle specific parsing for the 'multiply' tool input if string
        if action_name == "multiply" and isinstance(action_input, str):
            try:
                # Convert comma-separated string inputs to float numbers
                numbers = [float(x.strip()) for x in action_input.split(",")]
                result = self.tools[action_name].execute(*numbers)
            except Exception as e:
                result = f"Error parsing input for multiply: {e}"
        elif isinstance(action_input, dict):
            # Unpack dict inputs as keyword arguments
            result = self.tools[action_name].execute(**action_input)
        elif isinstance(action_input, (list, tuple)):
            # Unpack list/tuple inputs as positional arguments
            result = self.tools[action_name].execute(*action_input)
        else:
            # Pass input directly if none of above
            result = self.tools[action_name].execute(action_input)
        
        observation = f"Observation from {action_name}: {result}"
        print("observation from wiki tool: ", observation)
        # Add observation as a system message to conversation history
        self.messages.append({"role": "system", "content": observation})
        # Continue reasoning after receiving the observation
        self.think()
      else:
        # If requested action/tool not found, notify and continue thinking
        self.messages.append({"role": "system", "content": f"Unknown action: {action_name}"})
        self.think()

    # Entry point for running the agent with a user query
    def run(self,query):
        self.query =  query
        self.think()
        # Return the last message content if available, else None
        return self.messages[-1]["content"] if self.messages else None

# Main function demonstrating usage of the Agent class
def main():
    agent = Agent(max_iterations=2)  # Create agent with limited reasoning steps
    agent.load_template("./input_template.txt")  # Load prompt template
    
    # Import and register tools
    from tools.wiki import search
    from tools.multiply import multiply
    agent.register_tool("multiply", multiply)
    agent.register_tool("search", search)
    
    # Run the agent on a sample query
    query = "what's the product of 3,5?"
    agent.run(query)
    final_response = agent.final_answer
    print("Final response:", final_response)
    # print("Content: ", json.loads(response.content).get("action"))
    
# Run the main function when script is executed directly
if __name__ == '__main__':
    main()
