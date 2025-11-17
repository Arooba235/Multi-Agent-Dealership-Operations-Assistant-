# AutoAdvisor (LLM-Powered Car Recommendation System)
## Overview
AutoAdvisor is an intelligent, multi-agent system designed to help users make informed decisions when buying a car. The system uses a structured agent workflow—Inventory Analyst, Pricing Strategist, and Recommendation Agent—managed by a Supervisor (LangGraph). Each agent performs a specialized task, and the final output is a personalized car recommendation.The application includes a simple dashboard where you can update customer details (name, budget, max mileage, etc.). These inputs are stored locally and passed to the multi-agent reasoning pipeline. The final recommendation is generated using the Groq LLM.

## Tech Stack
- Backend: Flask, LangGraph to manage multi-agent workflow, Groq API for fast LLM inference
- Frontend: HTML, CSS
- Agents: 
    - Supervisor – Decides which agent should act next.
    - Inventory Analyst – Compares available inventory with customer preferences.
    - Pricing Strategist – Generates pricing insights and promotions.
    - Recommendation Agent – Produces the final recommendation using all previous outputs.