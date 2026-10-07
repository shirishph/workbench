import os
import pickle
import yaml
import pandas as pd
from openai import OpenAI


DATA_PATH = "../data/RO_Detail.csv"
MODEL_PATH = "../model.pkl"

YELLOW = "\033[93m"
LIGHT_BLUE = "\033[94m"
LIGHT_GREY = "\033[90m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"
UNDERLINE = "\033[4m"


# =========================================================
# PART 1 — Predicted Appointment Cap
# =========================================================

def predict(model, date, hour):
    return model.get((date.weekday(), hour))


# =========================================================
# PART 2 — AI Data Agent
# =========================================================

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

df_ro = pd.read_csv(DATA_PATH)

df_ro["created_at"] = pd.to_datetime(df_ro["created_at"])
df_ro["created_date"] = pd.to_datetime(df_ro["created_date"])


yaml_config_str = """
business_context:
  "This dataset contains dealership repair orders. The analysis focuses
   on customer service activity and appointment demand."

data_relationships:
  "created_at determines created_date and created_hour.
   created_date and created_hour describe when the repair order was created.
   appointment_at describes when the appointment is scheduled.
   These are different concepts.
   has_appointment identifies repair orders associated with appointments."

data_dictionary:
  df_ro:
    created_at:
      "The date and time when the repair order was created."
    created_date:
      "The calendar date when the repair order was created."
    created_hour:
      "The hour of the day when the repair order was created, from 0 through
      23. This is a time dimension for analyzing demand by hour. When a
      question asks for hourly demand, the hour must remain as a separate
      dimension in the result rather than being combined into one total."
    weekday:
      "The day of the week corresponding to created_date."
    has_appointment:
      "Indicates whether the repair order has an appointment."
    appointment_at:
      "The scheduled date and time of the appointment."
    service_mode:
      "The service mode associated with the repair order."
    department:
      "The service department associated with the repair order."
    advisor:
      "The service advisor associated with the repair order."
    tech_count:
      "The number of technicians associated with the repair order."
    job_count:
      "The number of jobs associated with the repair order."
    billed_hours:
      "The number of labor hours billed on the repair order."
    scheduled_hours:
      "The number of labor hours scheduled for the repair order."

data_definitions:
  hourly_appointment_demand:
    "Historical appointment demand is the number of repair orders with
     an appointment, grouped by the time when the repair order was created.

     The question may specify a particular date, a day of the week,
     or another time condition. Interpret that condition from the user's
     question and apply it to the appropriate date/time fields.

     If the question asks for hourly demand, preserve created_hour as
     the hour dimension in the result. Do not reduce multiple hours to
     one total.

     If the question specifies a weekday such as Monday, select records
     whose created_date falls on that weekday."

  appointment_demand_by_service_mode:
    "Appointment demand by service mode means the number of repair
     orders with an appointment, grouped according to their service mode."

  appointment_demand_by_department:
    "Analyze appointment demand by service department.

     When the question asks for a percentage, identify the population
     explicitly defined by the question.

     The phrase 'percentage of Monday appointments' means:
     - Population: all appointments on Monday.
     - Numerator: the subset of those Monday appointments satisfying
       the additional condition in the question.
     - Denominator: all Monday appointments.

     When the question asks for the percentage broken down by department,
     apply the same population definition separately within each department.

     For example, for:
     'What percentage of Monday appointments were created between
     8 AM and 10 AM, broken down by department?'

     the denominator for each department must be that department's
     total Monday appointments, not that department's appointments
     across all dates.

     Conditions describing the population must be applied to both
     numerator and denominator unless the question explicitly specifies
     otherwise.

     The numerator must always be a subset of the denominator."

  percentage_semantics:
    "When a question asks for a percentage, first identify the population
     described by the phrase following 'percentage of'.

     That population is the denominator.

     Apply all conditions that define the population before calculating
     the percentage.

     If the question asks for a percentage 'of Monday appointments',
     the denominator is the total number of Monday appointments.

     If the question asks for a percentage 'of Monday appointments
     created between 8 AM and 10 AM', interpret the full phrase
     carefully and distinguish the population from the subset being
     measured.

     When the question asks for a percentage broken down by a category,
     calculate the numerator and denominator at the same category
     level unless the question explicitly specifies otherwise.

     The numerator must be a subset of the denominator population.

     Do not use an unrestricted dataset, all-time total, or category
     total as the denominator when the question defines a narrower
     population."
"""


config = yaml.safe_load(yaml_config_str)

data_dictionary = config["data_dictionary"]
business_rules = config["data_definitions"]

schema_context = f"""
Business context:
{config["business_context"]}

Data relationships:
{config["data_relationships"]}

Available DataFrame:
`df_ro`

Column definitions:
{yaml.dump(data_dictionary["df_ro"])}

Sample rows:
{df_ro.head(3).to_dict(orient="records")}

examples:
  "If asked for hourly appointment demand, identify appointment repair
   orders and determine the number associated with the relevant date/hour.

   If asked for appointment demand by department, the result should
   retain the department breakdown rather than reducing all departments
   to one total.

   If asked for appointment demand by service mode, the result should
   retain the service-mode breakdown."
"""

# =========================================================
# Route question to YAML capability
# =========================================================

def choose_question(user_query):

    routing_prompt = f"""
You are a strict router for automotive service-demand
data analysis.

Available capabilities:

{yaml.dump(list(business_rules.keys()))}

User question:
{user_query}

Select the single capability whose underlying data and business
meaning can support answering the user's question.

A user question may require additional analytical operations using
an existing capability. These may include filtering, comparison,
ranking, percentages, averages, totals, or other calculations.

Do not require the question to exactly match the capability name
or description.

Choose the capability that provides the data needed to answer the
question.

Return ONLY the exact capability name.

If none of the capabilities provide the data needed to answer the
question, return exactly:

NONE
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": routing_prompt,
            }
        ],
        temperature=0.0,
    )

    rule_key = completion.choices[0].message.content.strip()

    if rule_key not in business_rules:
        return None

    return rule_key


# =========================================================
# Execute selected capability
# =========================================================

def execute_agent_query(user_query, rule_key, date, hour):

    applied_rule = business_rules[rule_key]

    """
    print(
        f"\nRouting request via YAML rule entry: "
        f"[{rule_key}]"
    )
    """
    print(f"{LIGHT_GREY}Routing request via YAML rule entry: [{rule_key}]{RESET}")


    system_prompt = f"""
You are a strict, precise Python code generator for
automotive service-demand data analytics.

{schema_context}

REQUESTED DATE: {date.date()}
REQUESTED HOUR: {hour}

BUSINESS RULE LOGIC:
{applied_rule}

PRESENTATION RULES:
- Return only the concise answer to the user's question.
- Start directly with the answer.
- Do not provide an introduction, preamble, explanation, or conclusion.
- Do not repeat or paraphrase the question.
- Do not describe the computation or how the answer was obtained.
- Do not use phrases such as "The analysis shows", "The results show",
  "The computational result shows", or similar result-introduction phrases.
- Preserve the meaning and values in the computational result.
- Use concise natural-language formatting appropriate to the result.
- For a simple categorical result, use:
  "Dept A: 98.53%. Dept B: 1.31%. Dept C: 0.16%."
- For a single result, use:
  "Dept A."
- For a result with a value, use:
  "Dept A: 98.76%."
- Percentages must use exactly two decimal places and the % symbol.

CRITICAL INSTRUCTIONS:

1. Write pure Python using pandas.
2. Do not import matplotlib or seaborn.
3. Compute the calculation requested by the business rule.
4. Assign the final calculation to `final_result`.
5. final_result must be a single numeric value.
6. For hourly_appointment_demand, return the appointment count
   for the requested date and hour only.
7. Do not estimate values.
8. Do not add explanations or conclusions to final_result.
9. Return ONLY executable Python code.
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        temperature=0.0,
    )

    generated_code = completion.choices[0].message.content.strip()

    if generated_code.startswith("```"):
        generated_code = generated_code.split("\n", 1)[1]
        generated_code = generated_code.rsplit("```", 1)[0].strip()    

    # print(f"\nGenerated Code:\n{generated_code}\n")
    print(f"{LIGHT_GREY}Generated Code:{RESET}")
    print(f"{LIGHT_GREY}{generated_code}{RESET}")

    try:

        runtime_scope = {
            "df_ro": df_ro,
            "pd": pd,
        }

        exec(
            generated_code,
            {},
            runtime_scope,
        )

        raw_result = runtime_scope.get("final_result")

        """
        print(
            f"Engine Executed Successfully."
            f"\nRaw output:\n{raw_result}"
        )
        """
        print(f"{LIGHT_GREY}Engine Executed Successfully.{RESET}")

        print(f"{LIGHT_GREY}Raw output:{RESET}")
        print(f"{LIGHT_GREY}{raw_result}{RESET}")

        synthesis = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a strict, evidence-grounded automotive
service-demand data analyst.

Use ONLY information contained in:

- the user question
- the business rule
- the computational result

Do not invent facts, causes, explanations,
recommendations, or numbers.

Preserve numerical values exactly.

State only what the analysis shows.

Keep the response concise and factual.

Return ONLY the final response.
"""
                },
                {
                    "role": "user",
                    "content": f"""
USER QUESTION:
{user_query}

BUSINESS RULE:
{applied_rule}

COMPUTATIONAL RESULT:
{raw_result}

Explain the computational result concisely.
"""
                },
            ],
            temperature=0.0,
        )

        answer = synthesis.choices[0].message.content

        prefixes = [
            "The computational result shows that ",
            "The computational result shows ",
            "The analysis shows that ",
            "The analysis shows ",
            "The results show that ",
            "The results show ",
        ]

        for prefix in prefixes:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip().capitalize()
                break

        print(
            f"\n{UNDERLINE}{CYAN}Query{RESET}: "
            f"{CYAN}{user_query}{RESET}"
        )

        print(
            f"{UNDERLINE}{GREEN}Answer{RESET}: "
            f"{GREEN}{answer}{RESET}\n"
        )

    except Exception as e:
        print(
            f"Error executing generated analysis: {e}"
        )


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Part 1
    # -----------------------------------------------------

    import sys

    if len(sys.argv) != 3:
        print("Usage: python run.py YYYY-MM-DD HH")
        sys.exit(1)

    try:
        date = pd.Timestamp(sys.argv[1])
        hour = int(sys.argv[2])
    except ValueError:
        print("Invalid date or hour.")
        sys.exit(1)

    if not 0 <= hour <= 23:
        print("Hour must be between 0 and 23.")
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    prediction = predict(model, date, hour)

    print("\n\n\n🔹 APPOINTMENT DEMAND FORECAST")
    print("=" * 50)
    print(LIGHT_BLUE)
    print(f"Date:                   {date.date()}")
    print(f"Hour:                   {hour:02d}:00")
    print(RESET)

    if prediction is None:
        print("Predicted appointment cap: N/A")
        print(
            "No historical data available "
            "for this weekday/hour."
        )
    else:
        print(YELLOW)
        print(
            f"Predicted appointment cap: "
            f"{prediction:.0f}"
        )
        print(RESET)

    # -----------------------------------------------------
    # Part 2
    # -----------------------------------------------------

    print("🔹 AI DATA AGENT")
    print("=" * 50)

    while True:

        print("\n\n\n\nPossible questions:")
        print("What was our hourly appointment demand?")
        print("Which department handled the most appointments?")
        print("How many appointments did Dept B have?")

        user_query = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if user_query.lower() == "exit":
            print("Exiting AI Data Agent.")
            break

        if not user_query:
            continue

        rule_key = choose_question(user_query)

        if rule_key is None:
            print(
                "\nI don't currently have an analysis "
                "capability for that question."
            )
            continue

        execute_agent_query(user_query, rule_key, date, hour)

