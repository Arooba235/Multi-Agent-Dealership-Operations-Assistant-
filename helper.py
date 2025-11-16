from typing_extensions import Literal, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, MessagesState
import random
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import pandas as pd
from ast import literal_eval
import json
from dotenv import load_dotenv
load_dotenv()

# os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_PROJECT"] = "testProject3"

llm = ChatGroq(model = "qwen/qwen3-32b")

class SupervisorState(MessagesState):
    """State for the multi-agent system"""
    next_agent:str = ""
    inventory_analyst: str = ""  
    pricing_strategist:str = ""
    recommendation:str = ""
    promotions: str = ""
    task_complete:bool = False
    current_task:str = ""

def create_supervisor_chain():
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """
    You are the Supervisor Agent managing three dealership agents:

    1. inventory_analyst - Reads inventory data and filters cars based on customer preferences.
    2. pricing_strategist - Applies pricing rules (markup, discounts, promotions) to shortlisted cars.
    3. recommendation - Selects the best car for the customer and justifies the choice.

    Your job is to control the workflow and decide **which agent should run next**.

    Current State:
    - Inventory shortlisted generated: {has_inventory}
    - Pricing analysis generated: {has_pricing}
    - Final recommendation generated: {has_recommendation}

    Rules:
    - If NO shortlist exists, run "inventory_analyst".
    - If shortlist exists BUT pricing not done, run "pricing_strategist".
    - If pricing done BUT final recommendation not generated, run "recommendation".
    - If all outputs exist, respond ONLY with "Done".

    Respond ONLY with one of these:
    - inventory_analyst
    - pricing_strategist
    - recommendation
    - Done
    """),
        ("human", "{task}")
    ])

    return supervisor_prompt | llm

def supervisor_agent(state: SupervisorState) -> Dict:
    """Supervisor decides next agent based on current dealership workflow."""
    
    messages = state["messages"]
    task = messages[-1].content if messages else "No task"

    has_inventory = bool(state.get("inventory_analyst", ""))
    has_pricing = bool(state.get("pricing_strategist", ""))
    has_recommendation = bool(state.get("recommendation", ""))

    # Supervisor chain
    chain = create_supervisor_chain()
    decision = chain.invoke({
        "task": task,
        "has_inventory": has_inventory,
        "has_pricing": has_pricing,
        "has_recommendation": has_recommendation
    })
    # Output
    decision_text = decision.content.strip().lower()
    decision_text = decision_text.split()[-1]

    # print("has_inventory", has_inventory)
    # print("has_pricing",has_pricing)
    # print("has_recommendation",has_recommendation)

    # Decide next agent
    if "done" in decision_text or has_recommendation:
        next_agent = "end"
        supervisor_msg = "Supervisor: All dealership tasks completed."
    
    elif "inventory_analyst" in decision_text or not has_inventory:
        next_agent = "inventory_analyst"
        supervisor_msg = "Supervisor: Starting with inventory filtering. Assigning to Inventory Analyst..."
    
    elif "pricing_strategist" in decision_text or (has_inventory and not has_pricing):
        next_agent = "pricing_strategist"
        supervisor_msg = "Supervisor: Inventory filtered. Applying pricing rules next..."
    
    elif "recommendation" in decision_text or (has_pricing and not has_recommendation):
        next_agent = "recommendation"
        supervisor_msg = "Supervisor: Pricing complete. Generating final customer recommendation..."
    
    else:
        next_agent = "end"
        supervisor_msg = "Supervisor: All tasks appear to be completed."

    return {
        "messages": [AIMessage(content=supervisor_msg)],
        "next_agent": next_agent,
        "current_task": task
    }

def inventory_analyst_agent(state: SupervisorState) -> Dict:
    """Inventory Analyst: reads inventory and filters based on customer.json preferences."""

    # customer preferences
    try:
        with open("customer.json", "r") as f:
            customer = json.load(f)
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error reading customer.json: {e}")],
            "inventory_analyst": "",
            "next_agent": "supervisor"
        }

    # inventory dataset
    try:
        df = pd.read_csv("cars.csv")
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error reading cars.csv: {e}")],
            "inventory_analyst": "",
            "next_agent": "supervisor"
        }

    # preference filters
    filtered = df.copy()
    
    if customer.get("budget"):
        filtered = filtered[filtered["price"] <= customer["budget"]]

    if customer.get("preferred_type"):
        filtered = filtered[
            filtered["type"].str.lower() == customer["preferred_type"].lower()
        ]

    if customer.get("max_mileage"):
        filtered = filtered[filtered["mileage"] <= customer["max_mileage"]]

    if customer.get("fuel_preference"):
        filtered = filtered[
            filtered["fuel_type"].str.lower() == customer["fuel_preference"].lower()
        ]

    shortlist = filtered.to_dict(orient="records")

    preview = shortlist[:3] if len(shortlist) > 0 else []

    agent_message = (
        "Inventory Analyst: Completed inventory filtering.\n\n"
        f"Customer Preferences:\n"
        f"- Budget: {customer['budget']}\n"
        f"- Type: {customer['preferred_type']}\n"
        f"- Max Mileage: {customer['max_mileage']}\n"
        f"- Fuel Type: {customer['fuel_preference']}\n\n"
        f"Shortlisted Cars: {len(shortlist)} match(es) found.\n\n"
        f"Preview:\n{json.dumps(preview, indent=2)}"
    )

    return {
        "messages": [AIMessage(content=agent_message)],
        "inventory_analyst": shortlist,
        "next_agent": "supervisor"
    }

def pricing_strategist_agent(state: SupervisorState) -> Dict:
    """Pricing Strategist Agent: Applies pricing rules to shortlisted cars."""

    shortlisted_cars = state.get("inventory_analyst", [])
    task = state.get("current_task", "Car Pricing Strategy")
    if state.get("promotions") == "":
        promotions_list = [
            ("Weekend sale", "-2%"),
            ("Holiday sale", "-5%"),
            ("New customer bonus", "-1%"),
            ("Clearance event", "-10%")
        ]

        selected = random.choice(promotions_list)
        promo_text = f"{selected[0]} ({selected[1]})"

        state["promotions"] = promo_text
    promotions = state.get("promotions", "")

    if not shortlisted_cars:
        return {
            "messages": [AIMessage(content="Pricing Strategist: No shortlisted cars found. Cannot apply pricing.")],
            "pricing_strategist": "",
            "next_agent": "supervisor"
        }

    # Create table
    car_list_text = "\n".join([
        f"- ID: {car['id']}, {car['make']} {car['model']} ({car['year']}), "
        f"Price: {car['price']}, Mileage: {car['mileage']} miles, "
        f"Popularity: {car['popularity_score']}, Days in Inventory: {car['days_in_inventory']}, "
        f"Condition: {car['condition']}"
        for car in shortlisted_cars
    ])

    # Prompt
    pricing_prompt = f"""
    You are a Pricing Strategy Agent in a car dealership.

    Apply the following pricing rules and compute final recommended prices.

    -----------------------------------
    DEALERSHIP PRICING RULES
    -----------------------------------

    1. MARKUP RULES (apply on base price):
    - Age < 3 years → +5%
    - Age 3-5 years → +3%
    - Age > 5 years → +1%
    - Popularity ≥ 8 → +2%

    2. DISCOUNTS:
    - Days > 90 → -7%
    - Days 60-90 → -5%
    - Days 30-60 → -3%
    - Popularity ≤ 5 → -3%

    3. CONDITION ADJUSTMENTS:
    - Excellent → +3%
    - Used → 0%
    - Fair → -5%
    - Needs Repair → -12%

    4. ACTIVE PROMOTIONS:
    \"\"\"{promotions}\"\"\"

    -----------------------------------
    INPUT SHORTLISTED CARS:
    {car_list_text}
    -----------------------------------

    Please return the final car prices in strict JSON format only, following this schema:

    [
        {{
            "car_id": <int>,
            "original_price": <number>,
            "final_recommended_price": <number>,
            "breakdown": {{
                "markup": "<percent>",
                "inventory_discount": "<percent>",
                "popularity_adjustment": "<percent>",
                "condition_adjustment": "<percent>",
                "promotion": "<percent>"
            }}
        }}
    ]

    Do NOT include any extra text, reasoning, or explanations. Strictly return the JSON.
    """

    # Invoke LLM
    response = llm.invoke([HumanMessage(content=pricing_prompt)])
    pricing_output = response.content
    agent_message = (
        "Pricing Strategist: Pricing rules applied successfully.\n"
        f"Sample Output Preview:\n{pricing_output[:400]}..."
    )
    return {
        "messages": [AIMessage(content=agent_message)],
        "pricing_strategist": pricing_output,
        "next_agent": "supervisor"
    }

def recommendation_agent(state: SupervisorState) -> Dict:
    """Final agent: uses LLM judgment to pick best car based on all previous outputs."""

    # customer preferences
    try:
        with open("customer.json", "r") as f:
            customer = json.load(f)
    except Exception as e:
        return {
            "messages": [AIMessage(content=f"Error reading customer.json: {e}")],
            "recommendation": "",
            "next_agent": "supervisor"
        }

    shortlist = state.get("inventory_analyst", [])
    pricing_output = state.get("pricing_strategist", "")

    if not shortlist:
        return {
            "messages": [AIMessage(content="Recommendation Agent: No shortlisted cars found.")],
            "recommendation": "",
            "next_agent": "supervisor"
        }

    # prompt
    recommendation_prompt = f"""
    You are the Customer Fit & Recommendation Agent.

    We have 3 sources of information:
    1. Customer Profile:
    {json.dumps(customer, indent=2)}

    2. Inventory Analyst Shortlist:
    {json.dumps(shortlist, indent=2)}

    3. Pricing Strategist Output (may include JSON + text):
    {pricing_output}

    Your task:

    - Evaluate the customer profile
    - Evaluate the shortlisted inventory cars
    - Use any useful pricing information found in the pricing strategist output
    - Select ONE best car overall
    - Provide a human-friendly justification explaining WHY this car is the best match.

    Rules:
    - DO NOT output JSON.
    - DO NOT list multiple cars.
    - Provide ONLY the final chosen car and a clear justification.

    Now give the final recommendation:
    """

    llm_response = llm.invoke([
        SystemMessage(content="You are an expert recommendation agent."),
        HumanMessage(content=recommendation_prompt)
    ])

    final_answer = llm_response.content.strip()

    return {
        "messages": [AIMessage(content=final_answer)],
        "recommendation": final_answer,
        "next_agent": "supervisor",
        "task_complete": True
    }

# Router Function
def router(state: SupervisorState) ->Literal['supervisor', 'inventory_analyst', 'pricing_strategist', 'recommendation', '__end__']:
    """Routes to next agent based on state"""
    next_agent = state.get("next_agent", "supervisor")
    if next_agent =="end" or state.get("task_complete",False):
        return END
    if next_agent in ["supervisor", "inventory_analyst", "pricing_strategist", "recommendation"]:
        return next_agent

    return "supervisor"

# Create workflow
workflow = StateGraph(SupervisorState)

# Add nodes
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("inventory_analyst", inventory_analyst_agent)
workflow.add_node("pricing_strategist", pricing_strategist_agent)
workflow.add_node("recommendation", recommendation_agent)

# Set entry point
workflow.set_entry_point("supervisor")
# Add routing
for node in ["supervisor", "inventory_analyst", "pricing_strategist", "recommendation"]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor":"supervisor",
            "inventory_analyst":"inventory_analyst",
            "pricing_strategist":"pricing_strategist",
            "recommendation":"recommendation",
            END:END 
        }
    )
graph = workflow.compile()