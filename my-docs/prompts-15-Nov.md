## Agent Loop PoC

1. I want you to write a poc for me in a new package : agent_loop_poc
2. It must demonstrate the working of agent loop in a  typical agentic workflow where in the following tool calls would be available : get_customer_by_name(string), get_customer_invoices_by_customer_id(int), calculate_invoices_total(list of invoices of the customer).
3. Implment these functions returning some random values 
4. Send these 3 functions as tools to the model with description and other details as needed
5. The model provider must be as same as the one used in : claude_subscription_poc
6. The agent must support the query like : Give me the list of invoices raised on John and the total of the invoices. 
7. The focus is on understanding the agent loop hence to print each iteration details - what goes to the model and what comes back from the model - specially whether it is a tool call or a end turn

## Agent loop with Langchain

1. Check the previous section : "Agent Loop PoC" and also the code that is already generated :  agent_loop_poc/agent_loop.py
2. Now build a new PoC in langchain_langgraph_poc implementating exact the same agent with same behavior but using langchain and langgraph.