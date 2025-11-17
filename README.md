# AutoAdvisor (LLM-Powered Car Recommendation System)
## Overview
AutoAdvisor is an intelligent, multi-agent system designed to help users make informed decisions when buying a car. The system uses a structured agent workflow—Inventory Analyst, Pricing Strategist, and Recommendation Agent—managed by a Supervisor (LangGraph). Each agent performs a specialized task, and the final output is a personalized car recommendation.The application includes a simple dashboard where you can update customer details (name, budget, max mileage, etc.). These inputs are stored locally and passed to the multi-agent reasoning pipeline. The final recommendation is generated using the Groq LLM.

## Tech Stack
- Backend: 
    - Flask
    - LangGraph to manage multi-agent workflow
    - Groq API for fast LLM inference
- Frontend: HTML, CSS
- Agents: 
    - Supervisor – Decides which agent should act next.
    - Inventory Analyst – Compares available inventory with customer preferences.
    - Pricing Strategist – Generates pricing insights and promotions.
    - Recommendation Agent – Produces the final recommendation using all previous outputs.
 
 ## Workflow
- User sends a query to the backend.
- Flask forwards the query to the LangGraph workflow.
- The Supervisor Agent decides which specialized agent to activate next based on current state:
    - Inventory Analyst: Filters cars based on customer preferences.
    - Pricing Strategist: Applies pricing rules and promotions to shortlisted cars.
    - Recommendation Agent: Selects the best car and provides justification.
- Each agent generates structured messages, which are sent back to the Supervisor.
- The Supervisor Agent combines all agent outputs and returns the final, comprehensive response to the user.
- Users can update customer preferences, which are saved and used by the agents for processing.

## Setup & Installation
- **Clone the Repository**
    ```bash
    git clone https://github.com/Arooba235/Multi-Agent-Dealership-Operations-Assistant-.git
    cd Multi-Agent-Dealership-Operations-Assistant-
    ```
- **Set Up Environment Variables**
    - Create a `.env` file in the project root.
    - Add your GROQ API key:
    ```
    GROQ_API_KEY=your_groq_api_key_here
    ```
- **Install Python dependencies**
    ```bash
    pip install -r requirements.txt
    ```
- **Run the Application**
    ```bash
    python app.py
    ```