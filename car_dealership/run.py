import io
import yaml
import pandas as pd
from openai import OpenAI

CYAN = "\033[96m"
GREEN = "\033[92m"
BOLD = "\033[1m"
RESET = "\033[0m"
UNDERLINE = "\033[4m"

# 1. Initialize OpenAI Client

# 2. Mock CSV Files (5 rows each, representing your 2 related data assets)
active_inventory_csv = """stock_id,model,days_in_inventory,shopper_leads_last_7_days,price,acquisition_type
STK001,Toyota Camry,15,4,24000,Purchase
STK002,Ford F-150,72,0,32000,Trade-in
STK003,Honda Civic,65,0,21000,Purchase
STK004,Jeep Wrangler,45,12,29000,Trade-in
STK005,BMW 3-Series,80,0,35000,Purchase"""

historical_sales_csv = """sale_id,stock_id,sale_price,profit_margin
SAL901,STK001,24500,12.5
SAL902,STK002,31000,5.0
SAL903,STK003,20500,14.0
SAL904,STK004,29500,8.5
SAL905,STK005,34000,4.5"""

# Load strings directly into pandas dataframes
df_inventory = pd.read_csv(io.StringIO(active_inventory_csv))
df_sales = pd.read_csv(io.StringIO(historical_sales_csv))

# 3. Define the Complete YAML Config Structure (Data Dictionary + Definitions)
yaml_config_str = """
data_dictionary:
  df_inventory:
    stock_id: "Unique text identifier for each car in stock (e.g., STK001)."
    model: "The make and model text name of the vehicle."
    days_in_inventory: "The number of days the vehicle has sat on the dealership lot unsold."
    shopper_leads_last_7_days: "The total count of customer inquiries or test drive requests received in the last week."
    price: "The list price of the vehicle."
    acquisition_type: "How the car was acquired. Contains only two values: 'Purchase' (bought from auction) or 'Trade-in' (swapped by a customer)."
  
  df_sales:
    stock_id: "Foreign key that connects back to the active inventory file stock_id."
    sale_price: "The absolute dollar amount the car sold for historically."
    profit_margin: "The net percentage profit made on the vehicle sale (e.g., 12.5 means 12.5%)."

data_definitions:
  vehicles_needing_attention: "Look at the active inventory file (df_inventory). Filter for rows where the vehicle has been in stock for more than 60 days AND it has received exactly 0 leads in the last 7 days. Return their stock_id and model."
  purchase_vs_trade: "Merge active inventory (df_inventory) and historical sales (df_sales) on stock_id. Group by acquisition_type and calculate the average profit_margin for each group."
"""

# Load configuration data structures
config = yaml.safe_load(yaml_config_str)
data_dictionary = config.get("data_dictionary", {})
business_rules = config.get("data_definitions", {})

# 4. Construct the prompt text using the yaml data mappings
schema_context = f"""
Available DataFrames in global memory with structural documentation:

1. `df_inventory`
   Columns Dictionary:
   {yaml.dump(data_dictionary.get('df_inventory'))}
   Sample Metadata: {df_inventory.head(2).to_dict(orient='records')}

2. `df_sales`
   Columns Dictionary:
   {yaml.dump(data_dictionary.get('df_sales'))}
   Sample Metadata: {df_sales.head(2).to_dict(orient='records')}
"""

def execute_agent_query(user_query, rule_key):
    applied_rule = business_rules.get(rule_key)
    
    system_prompt = f"""You are a strict, precise Python code generator for text-only automotive data analytics.
    {schema_context}

    BUSINESS RULE LOGIC TO ENFORCE IN CODE:
    {applied_rule}

    CRITICAL INSTRUCTIONS:
    1. Write pure Python code using pandas. Do NOT import matplotlib or seaborn. Avoid charting.
    2. Compute the calculation requested by the business rule logic above.
    3. The absolute final calculation result (string, dataframe conversion, dict) MUST be assigned to a variable named `final_result`.
    4. `final_result` MUST contain only values directly derived from the source data and the requested calculation. Do NOT add assumptions, interpretations, explanations, causes, predictions, or unsupported business conclusions.
    5. Do NOT infer relationships or causes that are not explicitly required by the business rule logic.
    6. Preserve the actual values produced by the calculation. Do not fabricate, estimate, or modify values.
    7. Return ONLY the raw code block text string. Do NOT enclose in markdown formatting backticks (```) or code markers.
    """

    print(f"\nRouting request via YAML rule entry: [{rule_key}]")
    
    # Generate Python instructions using GPT-4o-Mini
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.0
    )
    
    generated_code = completion.choices[0].message.content.strip()
    print(f"GPT-4o-Mini Code Output:\n{generated_code}\n")
    
    # 5. Local Sandbox Execution Environment
    try:
        # Pass variables directly to the interpreter runtime environment
        runtime_scope = {'df_inventory': df_inventory, 'df_sales': df_sales, 'pd': pd}
        exec(generated_code, {}, runtime_scope)
        
        # Scrape calculated computational payload back out from runtime environment memory
        raw_result = runtime_scope.get('final_result')
        print(f"Engine Executed Successfully. Raw output payload:\n{raw_result}")
        
        # Synthesize into clean markdown text 

        """
        synthesis = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"The user asked: '{user_query}'. The calculation engine produced this raw computational value: {raw_result}. Turn this into a concise textual business observation response. Do not use graphics or charts."}
            ],
            temperature=0.0
        )
        """

        synthesis = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a strict, evidence-grounded automotive data analyst.

Your job is to communicate the result of a completed data analysis.

CRITICAL INSTRUCTIONS:
1. Use ONLY information contained in the user question, business rule, and computational result.
2. Do NOT introduce facts, causes, explanations, assumptions, predictions, recommendations, or business implications that are not supported by these inputs.
3. Do NOT speculate about why a result occurred.
4. Do NOT infer facts merely because they seem plausible.
5. Do NOT use qualitative claims such as "significant", "concerning", "poor", "strong", "weak", "immediate", "strategic", or "better" unless directly supported by the computational result.
6. Preserve numerical values exactly as provided.
7. State what the analysis shows, not what might explain it.
8. If the evidence does not establish something, explicitly say that the available analysis does not establish it.
9. Keep the response concise and factual.
10. Return ONLY the final response to the user."""
                },
                {
                    "role": "user",
                    "content": f"""USER QUESTION:
        {user_query}

        BUSINESS RULE:
        {applied_rule}

        COMPUTATIONAL RESULT:
        {raw_result}

        Explain the computational result in a concise, evidence-grounded response."""
                }
            ],
            temperature=0.0
        )

        # print(f"\nQuery:\n{user_query}\n")
        # print(f"\nFinal Textual Response:\n{synthesis.choices[0].message.content}\n")
        print(f"\n{UNDERLINE}{CYAN}Query{RESET}: {CYAN}{user_query}{RESET}")
        print(
            f"{UNDERLINE}{GREEN}Answer{RESET}: "
            f"{GREEN}{synthesis.choices[0].message.content}{RESET}\n"
        )
                
    except Exception as e:
        print(f"❌ Error encountered running generated script string: {str(e)}")

# 6. Execute Testing Sequence
if __name__ == "__main__":
    # Test 1: Single file rule filtering logic
    execute_agent_query(
        user_query="Which vehicles need attention right now, and why?", 
        rule_key="vehicles_needing_attention"
    )
    
    print("="*80)
    
    # Test 2: Multi-file relational join calculations
    execute_agent_query(
        user_query="Are there meaningful differences between purchased vehicles vs trade-ins?", 
        rule_key="purchase_vs_trade"
    )

